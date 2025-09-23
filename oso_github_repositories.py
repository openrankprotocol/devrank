#!/usr/bin/env python3
"""
Crypto Ecosystem Contributor Network Analyzer - Pipeline Version

This script orchestrates a comprehensive crypto ecosystem analysis by using
two specialized modules in a multi-step pipeline:

1. Get seed organizations from config.toml and find all their repositories
2. Use repo_to_contributors.py to find core contributors of seed repos
3. Use contributors_to_repos.py to find extended organizations and repositories
4. Use repo_to_contributors.py again to find extended contributors

Prerequisites:
1. Create account at www.opensource.observer
2. Generate API key in Account Settings > API Keys
3. Set OSO_API_KEY environment variable or in .env file
4. Install dependencies: pip install pyoso pandas python-dotenv tomli
5. Configure parameters in config.toml

Usage:
    export OSO_API_KEY="your_api_key_here"
    python oso_github_repositories.py

Author: Open Source Observer Community
License: Apache 2.0
"""

import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path
import importlib.util

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import tomli
except ImportError:
    print("ERROR: tomli is required. Install with: pip install tomli")
    sys.exit(1)

# Import the two specialized modules
def import_module_from_path(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import the two analysis modules
try:
    repo_to_contributors = import_module_from_path("repo_to_contributors", "repo_to_contributors.py")
    contributors_to_repos = import_module_from_path("contributors_to_repos", "contributors_to_repos.py")
except Exception as e:
    print(f"ERROR: Failed to import analysis modules: {e}")
    print("Make sure repo_to_contributors.py and contributors_to_repos.py are in the same directory")
    sys.exit(1)


def load_config(config_path="config.toml"):
    """Load configuration from TOML file"""
    if not os.path.exists(config_path):
        print(f"ERROR: Configuration file {config_path} not found!")
        print("Please create config.toml file with your settings.")
        sys.exit(1)

    try:
        with open(config_path, "rb") as f:
            config = tomli.load(f)
        print(f"✓ Loaded configuration from {config_path}")
        return config
    except Exception as e:
        print(f"ERROR: Failed to load config file: {e}")
        sys.exit(1)


def validate_config(config):
    """Validate configuration"""
    required_sections = [
        "general", "seed_organizations", "seed_repositories",
        "contributors", "extended_repositories", "extended_contributors"
    ]

    for section in required_sections:
        if section not in config:
            print(f"ERROR: Missing required section [{section}] in config.toml")
            sys.exit(1)

    if not config["seed_organizations"]["orgs"]:
        print("ERROR: No seed organizations configured in config.toml!")
        sys.exit(1)

    return config


def discover_seed_repositories(config):
    """
    Step 1: Discover all repositories from seed organizations.

    Returns:
        list: List of repository identifiers in "org/repo" format
    """

    # Check if seed repositories file already exists
    output_dir = Path(config.get("general", {}).get("output_dir", "./raw"))
    seed_file = output_dir / "crypto_seed_repos.csv"

    if seed_file.exists():
        print("✓ Seed repositories file already exists, loading from file...")
        seed_repos_df = pd.read_csv(seed_file)
        seed_repos = seed_repos_df['repository_name'].tolist()
        print(f"✓ Loaded {len(seed_repos)} seed repositories from {seed_file}")
        return seed_repos

    # Get API key
    api_key = os.getenv('OSO_API_KEY')
    if not api_key:
        print("ERROR: OSO API key required. Set OSO_API_KEY environment variable.")
        sys.exit(1)

    # Initialize client
    try:
        import pyoso
        os.environ["OSO_API_KEY"] = api_key
        client = pyoso.Client()
        print("✓ OSO client initialized")
    except ImportError:
        print("ERROR: Install pyoso with: pip install pyoso pandas python-dotenv")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to initialize OSO client: {e}")
        sys.exit(1)

    seed_organizations = config["seed_organizations"]["orgs"]
    all_seed_repos = []

    print(f"Discovering repositories from {len(seed_organizations)} organizations...")

    for org in seed_organizations:
        print(f"  Finding repositories in {org}...")
        try:
            # Build exclude patterns filter
            exclude_conditions = []
            for pattern in config.get("seed_repositories", {}).get("exclude_patterns", []):
                exclude_conditions.append(f"LOWER(artifact_name) NOT LIKE LOWER('%{pattern}%')")
            exclude_filter = " AND " + " AND ".join(exclude_conditions) if exclude_conditions else ""

            # Query to find all repositories in organization
            org_repos_query = f"""
            SELECT
                artifact_namespace,
                artifact_name,
                CONCAT(artifact_namespace, '/', artifact_name) as repository_name
            FROM artifacts_by_project_v1
            WHERE artifact_source = 'GITHUB'
            AND artifact_namespace = '{org}'
            {exclude_filter}
            ORDER BY artifact_name
            LIMIT {config.get('query_limits', {}).get('seed_repos_per_org_limit', 100)}
            """

            org_repos_df = client.to_pandas(org_repos_query)

            if not org_repos_df.empty:
                org_repos = org_repos_df['repository_name'].tolist()
                all_seed_repos.extend(org_repos)
                print(f"    Found {len(org_repos)} repositories")
            else:
                print(f"    No repositories found for {org}")

        except Exception as e:
            print(f"    Error finding repositories for {org}: {e}")
            continue

    # Remove duplicates
    all_seed_repos = list(set(all_seed_repos))

    print(f"\n✓ Total seed repositories discovered: {len(all_seed_repos)} (after removing duplicates)")

    if not all_seed_repos:
        print("ERROR: No seed repositories found!")
        sys.exit(1)

    # Save seed repositories list
    output_dir = Path(config.get("general", {}).get("output_dir", "./raw"))
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_repos_df = pd.DataFrame({
        'repository_name': all_seed_repos,
    })

    seed_file = output_dir / "crypto_seed_repos.csv"
    seed_repos_df.to_csv(seed_file, index=False)
    print(f"✓ Saved seed repositories: {seed_file}")

    return all_seed_repos


def find_core_contributors(seed_repos, config):
    """
    Step 2: Find core contributors for seed repositories using repo_to_contributors.py

    Returns:
        list: List of core contributor handles
    """

    # Check if core contributors file already exists
    output_dir = Path(config.get("general", {}).get("output_dir", "./raw"))
    core_contrib_file = output_dir / "repo_contributors.csv"

    if core_contrib_file.exists():
        print("✓ Core contributors file already exists, loading from file...")
        core_contrib_df = pd.read_csv(core_contrib_file)
        core_contributors = core_contrib_df['contributor_handle'].unique().tolist()
        print(f"✓ Loaded {len(core_contributors)} core contributors from {core_contrib_file}")
        return core_contributors

    print(f"Finding contributors for {len(seed_repos)} seed repositories...")

    # Configure parameters from config.toml
    min_commits = config.get("contributors", {}).get("min_commits", 10)
    date_filter_days = config.get("general", {}).get("days_back", 0)

    # Use repo_to_contributors module
    contributors_df = repo_to_contributors.get_contributors_by_repos(
        repo_identifiers=seed_repos,
        min_commits=min_commits,
        date_filter_days=date_filter_days
    )

    if contributors_df.empty:
        print("ERROR: No contributors found for seed repositories")
        return []

    # Save contributors data
    saved_file = repo_to_contributors.save_contributors_data(contributors_df)

    # Get unique contributor handles, prioritizing top contributors
    contributor_summary = contributors_df.groupby('contributor_handle').agg({
        'total_commits': 'sum',
        'repository_name': 'nunique',
        'active_days': 'sum'
    }).sort_values('total_commits', ascending=False)

    # Apply configured limit for core contributors
    max_core_contributors = config.get("contributors", {}).get("max_core_contributors_for_extended_analysis", 30)
    top_contributors = contributor_summary.head(max_core_contributors)
    core_contributor_handles = top_contributors.index.tolist()

    return core_contributor_handles


def find_extended_ecosystem(core_contributors, config):
    """
    Step 3: Find extended organizations and repositories using contributors_to_repos.py

    Returns:
        tuple: (extended_repos_list, extended_organizations_list)
    """

    print(f"Finding repositories for {len(core_contributors)} core contributors...")

    # Configure parameters from config.toml
    min_commits = config.get("extended_repositories", {}).get("min_core_contributors", 2)
    include_org_repos = True  # Always include organization repos
    date_filter_days = config.get("general", {}).get("days_back", 0)

    # Use contributors_to_repos module
    repos_data = contributors_to_repos.get_repos_by_contributors(
        contributor_identifiers=core_contributors,
        min_commits=min_commits,
        include_org_repos=include_org_repos,
        date_filter_days=date_filter_days
    )

    contributed_repos_df = repos_data['contributed_repos']
    organization_repos_df = repos_data['organization_repos']

    if contributed_repos_df.empty and organization_repos_df.empty:
        print("ERROR: No extended repositories found")
        return [], []

    # Save repos data
    saved_files = contributors_to_repos.save_repos_data(repos_data)

    # Extract extended repositories list
    extended_repos = []
    extended_organizations = []

    if not contributed_repos_df.empty:
        # Get repositories that core contributors directly worked on
        contributed_repo_names = contributed_repos_df['repository_name'].unique().tolist()
        extended_repos.extend(contributed_repo_names)

        # Get organizations from contributed repos
        contributed_orgs = contributed_repos_df['organization'].unique().tolist()
        extended_organizations.extend(contributed_orgs)

    if not organization_repos_df.empty:
        # Get all repositories from the extended organizations
        org_repo_names = organization_repos_df['repository_name'].unique().tolist()

        # Apply limit to avoid overwhelming the system
        max_extended_repos = config.get("extended_repositories", {}).get("max_extended_repos", 200)
        if len(org_repo_names) > max_extended_repos:
            print(f"  Limiting extended repos to {max_extended_repos} (found {len(org_repo_names)})")
            org_repo_names = org_repo_names[:max_extended_repos]

        extended_repos.extend(org_repo_names)

        # Get unique organizations
        extended_orgs = organization_repos_df['organization'].unique().tolist()
        extended_organizations.extend(extended_orgs)

    # Remove duplicates and filter out seed repos if needed
    extended_repos = list(set(extended_repos))
    extended_organizations = list(set(extended_organizations))

    return extended_repos, extended_organizations


def find_extended_repos_by_stars(core_contributors, config, degree=1):
    """
    Find extended repos by stars from orgs that contributors work with

    Given contributors:
    1. Find all repos they contributed to
    2. Find all unique orgs/users that own these repos
    3. Take top 10 repos from each org/user ranked by stars

    Args:
        core_contributors: List of contributor handles
        config: Configuration dict
        degree: Degree of separation (1 or 2)

    Returns:
        list: List of extended repository names
    """
    # Get API key
    api_key = os.getenv('OSO_API_KEY')
    if not api_key:
        print("ERROR: OSO API key required. Set OSO_API_KEY environment variable.")
        sys.exit(1)

    # Initialize client
    try:
        import pyoso
        os.environ["OSO_API_KEY"] = api_key
        client = pyoso.Client()
        print("✓ OSO client initialized")
    except ImportError:
        print("ERROR: Install pyoso with: pip install pyoso pandas python-dotenv")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to initialize OSO client: {e}")
        sys.exit(1)

    if not core_contributors:
        print("ERROR: No core contributors provided")
        return []

    degree_text = "core" if degree == 1 else "extended"
    print(f"Finding repos contributed to by {len(core_contributors)} {degree_text} contributors...")

    # Build contributors filter
    contributors_str = "', '".join([c.replace("'", "''") for c in core_contributors])

    try:
        # Find all repos that core contributors worked on
        repos_query = f"""
        SELECT DISTINCT
            p.artifact_namespace as org_name,
            p.artifact_name as repo_name,
            CONCAT(p.artifact_namespace, '/', p.artifact_name) as repository_name
        FROM int_events_daily__github AS e
        JOIN int_github_users AS u ON e.from_artifact_id = u.artifact_id
        JOIN artifacts_by_project_v1 AS p ON e.to_artifact_id = p.artifact_id
        WHERE u.artifact_name IN ('{contributors_str}')
          AND e.event_type = 'COMMIT_CODE'
          AND u.artifact_name NOT LIKE '%[bot]'
          AND u.artifact_name NOT LIKE '%-bot'
          AND e.bucket_day >= DATE '2020-01-01'
          AND e.bucket_day <= DATE '2024-06-27'
          AND p.artifact_source = 'GITHUB'
        """

        repos_df = client.to_pandas(repos_query)

        if repos_df.empty:
            print("✗ No repositories found for core contributors")
            return []

        print(f"✓ Found {len(repos_df)} repositories contributed to by {degree_text} contributors")

        # Get unique organizations
        unique_orgs = repos_df['org_name'].unique()
        print(f"✓ Identified {len(unique_orgs)} unique organizations/users")

        # Process organizations in batches for faster queries
        extended_repos = []
        batch_size = 500

        for batch_start in range(0, len(unique_orgs), batch_size):
            batch_orgs = unique_orgs[batch_start:batch_start + batch_size]
            print(f"  Processing batch {batch_start // batch_size + 1}/{(len(unique_orgs) - 1) // batch_size + 1} ({len(batch_orgs)} organizations)...")

            try:
                # Build org conditions for batch
                org_conditions = []
                for org in batch_orgs:
                    org_conditions.append(f"p.artifact_namespace = '{org.replace(chr(39), chr(39)+chr(39))}'")

                org_condition_str = " OR ".join(org_conditions)

                stars_query = f"""
                WITH ranked_repos AS (
                    SELECT
                        p.artifact_namespace as org_name,
                        p.artifact_name as repo_name,
                        CONCAT(p.artifact_namespace, '/', p.artifact_name) as repository_name,
                        SUM(e.amount) as total_stars,
                        ROW_NUMBER() OVER (PARTITION BY p.artifact_namespace ORDER BY SUM(e.amount) DESC) as rn
                    FROM int_events_daily__github AS e
                    JOIN artifacts_by_project_v1 AS p ON e.to_artifact_id = p.artifact_id
                    WHERE ({org_condition_str})
                      AND e.event_type = 'STARRED'
                      AND p.artifact_source = 'GITHUB'
                    GROUP BY p.artifact_namespace, p.artifact_name
                )
                SELECT org_name, repo_name, repository_name, total_stars
                FROM ranked_repos
                WHERE rn <= 15
                ORDER BY org_name, total_stars DESC
                """

                batch_stars_df = client.to_pandas(stars_query)

                if not batch_stars_df.empty:
                    batch_repos = batch_stars_df['repository_name'].tolist()
                    extended_repos.extend(batch_repos)

                    # Show summary by org
                    org_counts = batch_stars_df.groupby('org_name').size()
                    for org, count in org_counts.items():
                        print(f"    {org}: Found {count} top starred repos")

            except Exception as e:
                print(f"    Error getting starred repos for batch: {e}")
                continue

        extended_repos = list(set(extended_repos))  # Remove duplicates

        # No limits - find as many repos as possible
        output_dir = Path(config.get("general", {}).get("output_dir", "./raw"))

        print(f"✓ Total extended repositories: {len(extended_repos)}")

        # Save extended repositories list
        output_dir.mkdir(parents=True, exist_ok=True)

        extended_repos_df = pd.DataFrame({
            'repository_name': extended_repos,
        })

        if degree == 1:
            extended_file = output_dir / "crypto_extended_repos_by_stars.csv"
            extended_repos_df.to_csv(extended_file, index=False)
            print(f"✓ Saved extended repositories: {extended_file}")
        else:
            extended_file = output_dir / "crypto_extended_repos_by_stars.csv"
            # Append to existing file
            if extended_file.exists():
                existing_df = pd.read_csv(extended_file)
                combined_df = pd.concat([existing_df, extended_repos_df], ignore_index=True)
                # Remove duplicates
                combined_df = combined_df.drop_duplicates(subset=['repository_name'])
                combined_df.to_csv(extended_file, index=False)
                print(f"✓ Appended {len(extended_repos_df)} repositories (total: {len(combined_df)}) to: {extended_file}")
            else:
                extended_repos_df.to_csv(extended_file, index=False)
                print(f"✓ Saved extended repositories: {extended_file}")

        return extended_repos

    except Exception as e:
        print(f"ERROR: Failed to find extended repos by stars: {e}")
        return []


def find_extended_contributors(extended_repos, config):
    """
    Step 4: Find extended contributors from extended repositories using repo_to_contributors.py

    Returns:
        pd.DataFrame: Extended contributors data
    """
    if not extended_repos:
        print("No extended repositories to analyze")
        return pd.DataFrame()

    print(f"Finding contributors for {len(extended_repos)} extended repositories...")

    # Configure parameters from config.toml
    min_commits = config.get("extended_contributors", {}).get("min_commits_extended", 3)
    date_filter_days = config.get("general", {}).get("days_back", 0)

    # Use repo_to_contributors module
    extended_contributors_df = repo_to_contributors.get_contributors_by_repos(
        repo_identifiers=extended_repos,
        min_commits=min_commits,
        date_filter_days=date_filter_days
    )

    if extended_contributors_df.empty:
        print("No extended contributors found")
        return pd.DataFrame()

    # No limits - keep all contributors
    filtered_extended_contributors = extended_contributors_df

    # Save extended contributors data with custom filename
    output_dir = Path(config.get("general", {}).get("output_dir", "./raw"))
    extended_contrib_file = output_dir / "crypto_extended_contributors_by_stars.csv"
    # Only save specified columns
    filtered_for_save = filtered_extended_contributors[['repository_name', 'contributor_handle']]

    # Check if this is second degree (append mode) - check if file already has data
    is_second_degree = extended_contrib_file.exists() and len(pd.read_csv(extended_contrib_file)) > 0

    if is_second_degree and extended_contrib_file.exists():
        # Append to existing file
        existing_df = pd.read_csv(extended_contrib_file)
        combined_df = pd.concat([existing_df, filtered_for_save], ignore_index=True)
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['repository_name', 'contributor_handle'])
        combined_df.to_csv(extended_contrib_file, index=False)
        print(f"✓ Appended {len(filtered_for_save)} contributors (total: {len(combined_df)}) to: {extended_contrib_file}")
    else:
        filtered_for_save.to_csv(extended_contrib_file, index=False)
        print(f"✓ Saved extended contributors: {extended_contrib_file}")

    return filtered_extended_contributors

def main(config_path="config.toml"):
    """Main orchestration function."""

    # Load and validate configuration
    config = load_config(config_path)
    config = validate_config(config)
    date_filter_days = config.get('general', {}).get('days_back', 0)
    date_filter_text = 'All time' if date_filter_days == 0 else f'{date_filter_days} days back'

    try:
        output_dir = Path(config.get("general", {}).get("output_dir", "./raw"))

        # Step 1: Discover seed repositories from organizations
        seed_repos_file = output_dir / "crypto_seed_repos.csv"
        if seed_repos_file.exists():
            print("✓ Step 1: Seed repositories file already exists, loading from file...")
            seed_repos_df = pd.read_csv(seed_repos_file)
            seed_repos = seed_repos_df['repository_name'].tolist()
            print(f"✓ Loaded {len(seed_repos)} seed repositories from {seed_repos_file}")
        else:
            seed_repos = discover_seed_repositories(config)

        # Step 2: Find core contributors using repo_to_contributors
        core_contrib_file = output_dir / "repo_contributors.csv"
        if core_contrib_file.exists():
            print("✓ Step 2: Core contributors file already exists, loading from file...")
            core_contrib_df = pd.read_csv(core_contrib_file)
            core_contributors = core_contrib_df['contributor_handle'].unique().tolist()
            print(f"✓ Loaded {len(core_contributors)} core contributors from {core_contrib_file}")
        else:
            core_contributors = find_core_contributors(seed_repos, config)

        # Step 3: Find extended repos by stars from orgs that core contributors work with
        extended_repos_file = output_dir / "crypto_extended_repos_by_stars.csv"
        if extended_repos_file.exists():
            print("✓ Step 3: Extended repos file already exists, loading from file...")
            extended_repos_df = pd.read_csv(extended_repos_file)
            extended_repos = extended_repos_df['repository_name'].tolist()
            print(f"✓ Loaded {len(extended_repos)} extended repositories from {extended_repos_file}")
        else:
            extended_repos = find_extended_repos_by_stars(core_contributors, config, degree=1)

        # Step 4: Find extended contributors using repo_to_contributors
        extended_contrib_file = output_dir / "crypto_extended_contributors_by_stars.csv"
        if extended_contrib_file.exists():
            print("✓ Step 4: Extended contributors file already exists, loading from file...")
            extended_contributors_df = pd.read_csv(extended_contrib_file)
            print(f"✓ Loaded {len(extended_contributors_df)} extended contributor records from {extended_contrib_file}")
        else:
            extended_contributors_df = find_extended_contributors(extended_repos, config)

        # Step 5 & 6: Second degree analysis (optional)
        enable_second_degree = config.get('general', {}).get('enable_second_degree_analysis', True)

        if enable_second_degree and not extended_contributors_df.empty:
            # Check if we have a second degree marker file to avoid re-running
            second_degree_marker = output_dir / ".second_degree_completed"

            if second_degree_marker.exists():
                print("✓ Step 5: Second degree analysis already completed, skipping...")
                extended_repos_2 = []
            else:
                extended_contributors_list = extended_contributors_df['contributor_handle'].unique().tolist()
                print(f"\nStarting second degree analysis with {len(extended_contributors_list)} contributors...")
                extended_repos_2 = find_extended_repos_by_stars(extended_contributors_list, config, degree=2)

            # Step 6: Find extended contributors second degree
            if extended_repos_2:
                print(f"Finding second degree contributors from {len(extended_repos_2)} repositories...")
                extended_contributors_2_df = find_extended_contributors(extended_repos_2, config)
                # Create marker file to indicate second degree completion
                second_degree_marker.touch()
                print("✓ Second degree analysis completed and marked")
            elif not second_degree_marker.exists():
                print("✓ Step 6: No second degree repos found")
        elif not enable_second_degree:
            print("✓ Second degree analysis disabled in configuration, skipping steps 5 & 6")
        else:
            print("✓ No extended contributors found, skipping second degree analysis")

    except KeyboardInterrupt:
        print("\n\n⏹️ Analysis cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        print("Full error trace:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        # Check for custom config file argument
        config_file = sys.argv[1] if len(sys.argv) > 1 else "config.toml"
        main(config_file)
    except KeyboardInterrupt:
        print("\nCancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Failed: {e}")
        sys.exit(1)
