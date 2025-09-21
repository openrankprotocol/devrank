#!/usr/bin/env python3
"""
Repository to Contributors Mapper

This script takes repository IDs and finds all contributors to those repositories
using the Open Source Observer (OSO) database.

Prerequisites:
1. Create account at www.opensource.observer
2. Generate API key in Account Settings > API Keys
3. Set OSO_API_KEY environment variable or in .env file
4. Install dependencies: pip install pyoso pandas python-dotenv

Usage:
    python repo_to_contributors.py
"""

import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_contributors_by_repos(repo_identifiers, min_commits=1, date_filter_days=0):
    """
    Find all contributors for given repositories.

    Args:
        repo_identifiers (list): List of repo identifiers in format ["org/repo", "org/repo"]
        min_commits (int): Minimum number of commits to be considered a contributor
        date_filter_days (int): Number of days back to consider (0 = all time)

    Returns:
        pd.DataFrame: Contributors data with columns:
            - repository_name: full repo name (org/repo)
            - contributor_handle: GitHub username
            - contributor_id: OSO artifact ID
            - total_commits: number of commits
            - active_days: number of days with activity
            - first_commit: first commit date
            - last_commit: last commit date
    """

    # Get API key
    api_key = os.getenv('OSO_API_KEY')
    if not api_key:
        print("ERROR: OSO API key required. Set OSO_API_KEY environment variable.")
        print("Get key at: https://www.opensource.observer")
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

    print(f"Finding contributors for {len(repo_identifiers)} repositories...")

    # Build date filter
    date_filter = ""
    if date_filter_days > 0:
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=date_filter_days)
        date_filter = f"AND e.bucket_day >= DATE '{cutoff_date.strftime('%Y-%m-%d')}'"

    # Process repositories in batches to avoid query compiler limit
    batch_size = 100
    all_contributors = []

    for i in range(0, len(repo_identifiers), batch_size):
        batch_repos = repo_identifiers[i:i+batch_size]
        print(f"  Processing batch {i//batch_size + 1}/{(len(repo_identifiers)-1)//batch_size + 1} ({len(batch_repos)} repositories)...")

        # Build repository conditions for this batch
        repo_conditions = []
        for repo_id in batch_repos:
            if '/' in repo_id:
                org, repo = repo_id.split('/', 1)
                repo_conditions.append(f"(p.artifact_namespace = '{org}' AND p.artifact_name = '{repo}')")
            else:
                print(f"WARNING: Invalid repo format '{repo_id}', should be 'org/repo'")

        if not repo_conditions:
            continue

        repo_condition_str = " OR ".join(repo_conditions)

        try:
            contributors_query = f"""
            SELECT
                CONCAT(p.artifact_namespace, '/', p.artifact_name) as repository_name,
                u.artifact_name as contributor_handle,
                u.artifact_id as contributor_id,
                SUM(e.amount) as total_commits,
                COUNT(DISTINCT e.bucket_day) as active_days,
                MIN(e.bucket_day) as first_commit,
                MAX(e.bucket_day) as last_commit
            FROM int_events_daily__github AS e
            JOIN int_github_users AS u
              ON e.from_artifact_id = u.artifact_id
            JOIN artifacts_by_project_v1 AS p
              ON e.to_artifact_id = p.artifact_id
              AND p.artifact_source = 'GITHUB'
            WHERE
              e.event_type = 'COMMIT_CODE'
              AND ({repo_condition_str})
              AND u.artifact_name IS NOT NULL
              AND u.artifact_name != ''
              {date_filter}
            GROUP BY p.artifact_namespace, p.artifact_name, u.artifact_name, u.artifact_id
            HAVING SUM(e.amount) >= {min_commits}
            ORDER BY p.artifact_namespace, p.artifact_name, SUM(e.amount) DESC
            """

            batch_contributors_df = client.to_pandas(contributors_query)

            if not batch_contributors_df.empty:
                all_contributors.append(batch_contributors_df)
                print(f"    Found {len(batch_contributors_df)} contributor records in this batch")

        except Exception as e:
            print(f"    ERROR: Batch query failed: {str(e)}")
            continue

    # Combine all batches
    if all_contributors:
        contributors_df = pd.concat(all_contributors, ignore_index=True)
        print(f"✓ Found {len(contributors_df)} contributor records")
        return contributors_df
    else:
        print("✗ No contributors found for the specified repositories")
        return pd.DataFrame()


def save_contributors_data(contributors_df, output_dir="./raw"):
    """Save contributors data to CSV file."""

    if contributors_df.empty:
        print("No data to save")
        return None

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save to CSV (always overwrite, no timestamp)
    csv_file = output_path / "repo_contributors.csv"
    # Only save specified columns
    filtered_df = contributors_df[['repository_name', 'contributor_handle']]
    filtered_df.to_csv(csv_file, index=False)

    return str(csv_file)
