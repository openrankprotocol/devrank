#!/usr/bin/env python3
"""
Temporary script to check if repositories exist in different OSO database tables.
Checks artifacts_by_project_v1, artifacts_v1, and int_artifacts__github tables.
"""

import os
import sys
import argparse
import tomli
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict

def check_repo_by_artifact_source_id(artifact_source_id, client):
    """
    Check if a repository exists in different OSO tables by artifact_source_id.
    Returns dict with table_name -> list of (artifact_id, org/repo) mappings.
    """
    results = {}

    tables = [
        ("artifacts_by_project_v1", "artifact_source"),
        ("artifacts_v1", "artifact_source"),
        ("int_artifacts__github", None)
    ]

    for table_name, extra_field in tables:
        try:
            extra_select = f", {extra_field}" if extra_field else ""
            query = f"""
            SELECT
                artifact_id,
                artifact_namespace,
                artifact_name{extra_select}
            FROM {table_name}
            WHERE artifact_source_id = '{artifact_source_id}'
            """

            df = client.to_pandas(query)

            if not df.empty:
                results[table_name] = []
                for _, row in df.iterrows():
                    org_repo = f"{row['artifact_namespace']}/{row['artifact_name']}"
                    results[table_name].append((row['artifact_id'], org_repo))
        except Exception:
            pass

    return results


def check_repo_in_tables(repo_name, client, artifact_tracking):
    """
    Check if a repository exists in different OSO tables.
    Returns list of artifact_source_ids found.
    """
    if '/' not in repo_name:
        return []

    org, repo = repo_name.split('/', 1)
    found_artifact_source_ids = []

    tables = [
        ("artifacts_by_project_v1", "GITHUB"),
        ("artifacts_v1", "GITHUB"),
        ("int_artifacts__github", None)
    ]

    tables_found = []

    for table_name, source_filter in tables:
        source_condition = f"AND artifact_source = '{source_filter}'" if source_filter else ""
        query = f"""
        SELECT artifact_source_id
        FROM {table_name}
        WHERE artifact_namespace = '{org}'
          AND artifact_name = '{repo}'
          {source_condition}
        """

        df = client.to_pandas(query)

        if not df.empty:
            tables_found.append(table_name)
            for _, row in df.iterrows():
                if row['artifact_source_id'] not in found_artifact_source_ids:
                    found_artifact_source_ids.append(row['artifact_source_id'])

    # Track results for each artifact_source_id found
    for artifact_source_id in found_artifact_source_ids:
        results = check_repo_by_artifact_source_id(artifact_source_id, client)
        if artifact_source_id not in artifact_tracking:
            artifact_tracking[artifact_source_id] = {}

        for table_name, artifacts in results.items():
            if table_name not in artifact_tracking[artifact_source_id]:
                artifact_tracking[artifact_source_id][table_name] = []
            artifact_tracking[artifact_source_id][table_name].extend(artifacts)

    status = "✅" if tables_found else "❌"
    tables_str = ", ".join(tables_found) if tables_found else "none"
    print(f"{status} {repo_name} -> {tables_str}")

    return found_artifact_source_ids


def main():
    """Main function to check repositories in OSO tables."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Check if repositories exist in OSO database tables')
    parser.add_argument('config', nargs='?', default='config.toml', help='Path to config.toml file (default: config.toml)')
    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    # Load configuration from TOML file
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"❌ ERROR: Configuration file {config_path} not found!")
        print(f"Usage: python check_repo_tables.py --config {config_path}")
        sys.exit(1)

    try:
        with open(config_path, "rb") as f:
            config = tomli.load(f)
        print(f"✅ Loaded configuration from {config_path}")
    except Exception as e:
        print(f"❌ ERROR: Failed to load configuration: {e}")
        sys.exit(1)

    # Get seed repositories from config
    seed_repos = config.get("general", {}).get("seed_repos", [])
    if not seed_repos:
        print("❌ ERROR: No seed repositories found in configuration!")
        print("Please ensure your config.toml has seed_repos defined in [general] section.")
        sys.exit(1)

    # Get API key
    api_key = os.getenv('OSO_API_KEY')
    if not api_key:
        print("❌ ERROR: OSO API key required. Set OSO_API_KEY environment variable.")
        sys.exit(1)

    # Initialize OSO client
    try:
        from pyoso import Client
        client = Client(api_key=api_key)
        print("✅ OSO client initialized successfully")
    except ImportError:
        print("❌ ERROR: pyoso library not found. Install with: pip install pyoso")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize OSO client: {e}")
        sys.exit(1)

    print("🚀 Starting repository table checks...")
    print(f"📊 Will check {len(seed_repos)} repositories from {config_path}")

    # Track artifacts by source_id
    artifact_tracking = defaultdict(dict)

    # Create job queue for retry logic
    repo_queue = seed_repos.copy()
    processed_repos = 0

    # Process repos in queue with retry logic
    while repo_queue:
        import time
        time.sleep(0.1)  # Add small delay to avoid rate limiting

        # Take one repo from queue
        repo = repo_queue.pop(0)

        try:
            check_repo_in_tables(repo, client, artifact_tracking)
            processed_repos += 1
        except Exception as e:
            # Handle exceptions that may have JSON parsing issues when converted to string
            try:
                error_msg = str(e)
            except Exception:
                error_msg = f"{type(e).__name__}: Unable to parse error message"
            print(f"❌ ERROR: Query for {repo} failed: {error_msg}")
            # Add failed repo back to the end of the queue for retry
            print(f"    Adding {repo} back to queue for retry")
            repo_queue.append(repo)
            continue

    print("\n" + "=" * 60)
    print("✅ Repository table checks completed!")

    # Find artifact_source_ids with multiple artifacts
    print("\n🔍 Artifact source IDs with multiple artifacts found:")
    duplicates_found = False

    for artifact_source_id, table_results in artifact_tracking.items():
        total_artifacts = 0
        all_repos = set()

        for table_name, artifacts in table_results.items():
            total_artifacts += len(artifacts)
            for artifact_id, org_repo in artifacts:
                all_repos.add(org_repo)

        if total_artifacts > 1:
            duplicates_found = True
            print(f"\n📍 {artifact_source_id}")
            print(f"   Total artifacts: {total_artifacts}")
            print(f"   Repositories: {', '.join(sorted(all_repos))}")
            for table_name, artifacts in table_results.items():
                if artifacts:
                    artifact_ids = [str(aid) for aid, _ in artifacts]
                    print(f"   {table_name}: {len(artifacts)} artifacts (IDs: {', '.join(artifact_ids)})")

    if not duplicates_found:
        print("   None found - all artifact_source_ids have single artifacts.")


if __name__ == "__main__":
    main()
