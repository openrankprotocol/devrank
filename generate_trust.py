#!/usr/bin/env python3
"""
Generate Local Trust Relationships from OSO Database (Using New Pipeline Files)

This script uses the new pipeline files (crypto_extended_contributors_by_stars.csv)
to identify user-repo pairs, then queries OSO database for those specific pairs.
"""

import os
import sys
import pandas as pd
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def generate_trust_relationships():
    """Generate trust relationships using raw file connections + OSO queries."""

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

    raw_dir = Path("raw")
    trust_dir = Path("trust")

    # Create trust directory if it doesn't exist
    trust_dir.mkdir(exist_ok=True)

    if not raw_dir.exists():
        print(f"ERROR: {raw_dir} directory not found!")
        return

    print("Loading existing connections from raw files...")

    # Collect all user-repo pairs from existing data
    user_repo_pairs = set()
    all_repos = set()

    # Load all repositories from seed and extended repos files
    seed_repos_file = raw_dir / "crypto_seed_repos.csv"
    if seed_repos_file.exists():
        df = pd.read_csv(seed_repos_file)
        seed_repos = set(df['repository_name'].dropna().tolist())
        all_repos.update(seed_repos)
        print(f"  Loaded {len(seed_repos)} seed repositories")

    extended_repos_file = raw_dir / "crypto_extended_repos_by_stars.csv"
    if extended_repos_file.exists():
        df = pd.read_csv(extended_repos_file)
        extended_repos = set(df['repository_name'].dropna().tolist())
        all_repos.update(extended_repos)
        print(f"  Loaded {len(extended_repos)} extended repositories by stars")

    print(f"  Total unique repositories: {len(all_repos)}")

    # From repo_contributors.csv
    repo_contrib_file = raw_dir / "repo_contributors.csv"
    if repo_contrib_file.exists():
        df = pd.read_csv(repo_contrib_file)
        for _, row in df.iterrows():
            user = row['contributor_handle']
            repo = row['repository_name']
            if pd.notna(user) and pd.notna(repo) and repo in all_repos:
                user_repo_pairs.add((user, repo))
        print(f"  Found {len(df)} pairs from repo_contributors.csv (filtered to our repos)")

    # From crypto_extended_contributors_by_stars.csv (has both columns)
    extended_contrib_file = raw_dir / "crypto_extended_contributors_by_stars.csv"
    if extended_contrib_file.exists():
        df = pd.read_csv(extended_contrib_file)
        for _, row in df.iterrows():
            user = row['contributor_handle']
            repo = row['repository_name']
            if pd.notna(user) and pd.notna(repo) and repo in all_repos:
                user_repo_pairs.add((user, repo))
        print(f"  Found {len(df)} pairs from crypto_extended_contributors_by_stars.csv")

    print(f"Total unique user-repo pairs: {len(user_repo_pairs)}")

    if not user_repo_pairs:
        print("ERROR: No user-repo pairs found in raw data!")
        return

    # Convert to list and process in batches
    pairs_list = list(user_repo_pairs)
    batch_size = 500  # Small batches to avoid query limits

    all_trust_dfs = []

    print("Querying OSO database for specific user-repo pairs...")

    # Initialize output file
    output_file = trust_dir / "github.csv"
    batch_results = []

    try:
        for batch_start in range(0, len(pairs_list), batch_size):
            batch_pairs = pairs_list[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(pairs_list) - 1) // batch_size + 1
            print(f"  Processing batch {batch_num}/{total_batches} ({len(batch_pairs)} pairs)")

            # Extract users and repos from this batch
            users_in_batch = set()
            repos_in_batch = set()

            for user, repo in batch_pairs:
                users_in_batch.add(user.replace("'", "''"))
                repos_in_batch.add(repo)

            users_str = "', '".join(users_in_batch)

            # Build repo conditions
            repo_conditions = []
            for repo in repos_in_batch:
                if '/' in repo:
                    org, repo_name = repo.split('/', 1)
                    repo_conditions.append(f"(p.artifact_namespace = '{org}' AND p.artifact_name = '{repo_name}')")

            if not repo_conditions:
                continue

            repo_condition_str = " OR ".join(repo_conditions)

            # User-to-repo trust query with proper weights (no age factor for now)
            user_to_repo_query = f"""
            SELECT
                u.artifact_name AS i,
                CONCAT(p.artifact_namespace, '/', p.artifact_name) AS j,
                SUM(
                    e.amount *
                    CASE
                        WHEN e.event_type = 'FORKED' THEN 1
                        WHEN e.event_type = 'STARRED' THEN 5
                        WHEN e.event_type = 'ISSUE_OPENED' THEN 10
                        WHEN e.event_type = 'PULL_REQUEST_OPENED' THEN 20
                        WHEN e.event_type = 'PULL_REQUEST_MERGED' THEN 10
                        WHEN e.event_type = 'COMMIT_CODE' THEN 5
                        ELSE 0
                    END
                ) AS v
            FROM int_events_daily__github AS e
            JOIN int_github_users AS u ON e.from_artifact_id = u.artifact_id
            JOIN artifacts_by_project_v1 AS p ON e.to_artifact_id = p.artifact_id
            WHERE u.artifact_name IN ('{users_str}')
              AND ({repo_condition_str})
              AND e.event_type IN ('FORKED', 'STARRED', 'ISSUE_OPENED', 'PULL_REQUEST_OPENED', 'PULL_REQUEST_MERGED', 'COMMIT_CODE')
              AND u.artifact_name NOT LIKE '%[bot]'
              AND u.artifact_name NOT LIKE '%-bot'
              AND u.artifact_name != 'safe-infra'
              AND e.bucket_day >= DATE '1980-01-01'
              AND e.bucket_day <= DATE '2024-06-27'
              AND p.artifact_source = 'GITHUB'
            GROUP BY u.artifact_name, CONCAT(p.artifact_namespace, '/', p.artifact_name)
            HAVING SUM(e.amount) > 0
            """

            try:
                batch_df = client.to_pandas(user_to_repo_query)
                if not batch_df.empty:
                    batch_results.append(batch_df)
                    print(f"    Found {len(batch_df)} user-to-repo relationships")
            except Exception as e:
                print(f"    Error in user-to-repo batch: {e}")

            # Repo-to-user credit query with proper weights (no age factor for now)
            repo_to_user_query = f"""
            SELECT
                CONCAT(p.artifact_namespace, '/', p.artifact_name) AS i,
                u.artifact_name AS j,
                SUM(
                    e.amount *
                    CASE
                        WHEN e.event_type = 'PULL_REQUEST_OPENED' THEN 5
                        WHEN e.event_type = 'PULL_REQUEST_MERGED' THEN 1
                        WHEN e.event_type = 'COMMIT_CODE' THEN 3
                        ELSE 0
                    END
                ) AS v
            FROM int_events_daily__github AS e
            JOIN int_github_users AS u ON e.from_artifact_id = u.artifact_id
            JOIN artifacts_by_project_v1 AS p ON e.to_artifact_id = p.artifact_id
            WHERE u.artifact_name IN ('{users_str}')
              AND ({repo_condition_str})
              AND e.event_type IN ('PULL_REQUEST_OPENED', 'PULL_REQUEST_MERGED', 'COMMIT_CODE')
              AND u.artifact_name NOT LIKE '%[bot]'
              AND u.artifact_name NOT LIKE '%-bot'
              AND u.artifact_name != 'safe-infra'
              AND e.bucket_day >= DATE '1980-01-01'
              AND e.bucket_day <= DATE '2024-06-27'
              AND p.artifact_source = 'GITHUB'
            GROUP BY CONCAT(p.artifact_namespace, '/', p.artifact_name), u.artifact_name
            HAVING SUM(e.amount) > 0
            """

            try:
                batch_df = client.to_pandas(repo_to_user_query)
                if not batch_df.empty:
                    batch_results.append(batch_df)
                    print(f"    Found {len(batch_df)} repo-to-user relationships")
            except Exception as e:
                print(f"    Error in repo-to-user batch: {e}")

            # Save every 10 batches to avoid data loss
            if batch_num % 10 == 0 or batch_num == total_batches:
                if batch_results:
                    current_df = pd.concat(batch_results, ignore_index=True)
                    # Round trust values
                    current_df['v'] = current_df['v'].round(6)

                    # Append to file (create with header if first save)
                    if batch_num == 10 or not output_file.exists():
                        current_df.to_csv(output_file, index=False, mode='w')
                        print(f"    ✓ Saved {len(current_df)} relationships to {output_file} (batch {batch_num})")
                    else:
                        current_df.to_csv(output_file, index=False, mode='a', header=False)
                        print(f"    ✓ Appended {len(current_df)} relationships to {output_file} (batch {batch_num})")

                    batch_results = []  # Clear batch results after saving

        print(f"\n✓ Processing complete! All results saved to {output_file}")

    except Exception as e:
        print(f"ERROR: Processing failed: {e}")
        return

if __name__ == "__main__":
    generate_trust_relationships()
