#!/usr/bin/env python3
"""
Trust Relationship Generator

This script calculates trust relationships between contributors based on
their shared repository contributions using the Open Source Observer (OSO) database.

Prerequisites:
1. Create account at www.opensource.observer
2. Generate API key in Account Settings > API Keys
3. Set OSO_API_KEY environment variable or in .env file
4. Install dependencies: pip install pyoso pandas python-dotenv

Usage:
    python generate_trust.py
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

def build_filter_conditions(config):
    """Build SQL filter conditions from config.toml filters"""
    filters = config.get("filters", {})

    # Bot filtering
    bot_conditions = []
    if filters.get("exclude_bots", True):
        bot_keywords = filters.get("bot_keywords", ["bot", "dependabot", "mergify", "renovate", "github-actions", "semantic-release"])
        for keyword in bot_keywords:
            bot_conditions.extend([
                f"u.artifact_name NOT LIKE '%{keyword}%'",
                f"u.artifact_name NOT LIKE '%[{keyword}]%'"
            ])
        # Add generic bot patterns
        bot_conditions.extend([
            "u.artifact_name NOT LIKE '%[bot]'",
            "u.artifact_name NOT LIKE '%-bot'"
        ])

    return {
        'bot_filter': ' AND '.join(bot_conditions) if bot_conditions else '',
    }


def build_user_to_repo_query(users_str, repo_condition_str, date_filter="", bot_filter=""):
    """Build simplified user-to-repo trust query."""
    return f"""
    SELECT
        u.artifact_name AS i,
        CONCAT(p.artifact_namespace, '/', p.artifact_name) AS j,
        SUM(
            e.amount *
            CASE
                WHEN e.event_type = 'COMMIT_CODE' THEN 5
                WHEN e.event_type = 'PULL_REQUEST_OPENED' THEN 20
                WHEN e.event_type = 'PULL_REQUEST_MERGED' THEN 10
                WHEN e.event_type = 'STARRED' THEN 5
                WHEN e.event_type = 'ISSUE_OPENED' THEN 10
                WHEN e.event_type = 'FORKED' THEN 1
                ELSE 0
            END
        ) AS v
    FROM int_events_daily__github AS e
    JOIN int_github_users AS u ON e.from_artifact_id = u.artifact_id
    JOIN artifacts_by_project_v1 AS p ON e.to_artifact_id = p.artifact_id
    WHERE u.artifact_name IN ('{users_str}')
      AND ({repo_condition_str})
      AND e.event_type IN ('COMMIT_CODE', 'PULL_REQUEST_OPENED', 'PULL_REQUEST_MERGED', 'STARRED', 'ISSUE_OPENED', 'FORKED')
      {bot_filter}
      AND p.artifact_source = 'GITHUB'
      {date_filter}
    GROUP BY u.artifact_name, CONCAT(p.artifact_namespace, '/', p.artifact_name)
    HAVING SUM(e.amount) > 0
    """

def build_repo_to_user_query(users_str, repo_condition_str, date_filter="", bot_filter=""):
    """Build simplified repo-to-user trust query."""
    return f"""
    SELECT
        CONCAT(p.artifact_namespace, '/', p.artifact_name) AS i,
        u.artifact_name AS j,
        SUM(
            e.amount *
            CASE
                WHEN e.event_type = 'COMMIT_CODE' THEN 3
                WHEN e.event_type = 'PULL_REQUEST_OPENED' THEN 5
                WHEN e.event_type = 'PULL_REQUEST_MERGED' THEN 1
                ELSE 0
            END
        ) AS v
    FROM int_events_daily__github AS e
    JOIN int_github_users AS u ON e.from_artifact_id = u.artifact_id
    JOIN artifacts_by_project_v1 AS p ON e.to_artifact_id = p.artifact_id
    WHERE u.artifact_name IN ('{users_str}')
      AND ({repo_condition_str})
      AND e.event_type IN ('COMMIT_CODE', 'PULL_REQUEST_OPENED', 'PULL_REQUEST_MERGED')
      {bot_filter}
      AND p.artifact_source = 'GITHUB'
      {date_filter}
    GROUP BY CONCAT(p.artifact_namespace, '/', p.artifact_name), u.artifact_name
    HAVING SUM(e.amount) > 0
    """

def load_user_repo_pairs():
    """Load user-repo pairs efficiently from CSV files."""
    print("Loading user-repo pairs from raw files...")

    raw_dir = Path("raw")
    if not raw_dir.exists():
        print(f"ERROR: {raw_dir} directory not found!")
        return set(), set()

    user_repo_pairs = set()
    all_repos = set()

    # Load repositories efficiently using pandas
    for file_name, repo_col in [
        ("crypto_seed_repos.csv", "repository_name"),
        ("crypto_extended_repos_by_stars.csv", "repository_name")
    ]:
        file_path = raw_dir / file_name
        if file_path.exists():
            df = pd.read_csv(file_path, usecols=[repo_col])
            repos = set(df[repo_col].dropna().tolist())
            all_repos.update(repos)
            print(f"  Loaded {len(repos)} repositories from {file_name}")

    print(f"  Total unique repositories: {len(all_repos)}")

    # Load user-repo pairs efficiently
    for file_name, user_col, repo_col in [
        ("repo_contributors.csv", "contributor_handle", "repository_name"),
        ("crypto_extended_contributors_by_stars.csv", "contributor_handle", "repository_name")
    ]:
        file_path = raw_dir / file_name
        if file_path.exists():
            df = pd.read_csv(file_path, usecols=[user_col, repo_col])
            # Vectorized filtering
            df = df.dropna()
            df = df[df[repo_col].isin(all_repos)]

            pairs = set(zip(df[user_col], df[repo_col]))
            user_repo_pairs.update(pairs)
            print(f"  Loaded {len(pairs)} pairs from {file_name}")

    print(f"Total unique user-repo pairs: {len(user_repo_pairs)}")
    return user_repo_pairs, all_repos

def process_batch_optimized(batch_pairs, client, date_filter="", config=None):
    """Process a batch of user-repo pairs with separate simpler queries."""
    import time

    # Build filter conditions from config
    filter_conditions = build_filter_conditions(config or {})
    bot_filter = f"AND {filter_conditions['bot_filter']}" if filter_conditions['bot_filter'] else ""

    # Extract users and repos from batch
    users_in_batch = set()
    repos_in_batch = set()

    for user, repo in batch_pairs:
        users_in_batch.add(user.replace("'", "''"))
        repos_in_batch.add(repo)

    users_str = "', '".join(users_in_batch)

    # Build repo conditions more efficiently
    repo_conditions = []
    for repo in repos_in_batch:
        if '/' in repo:
            org, repo_name = repo.split('/', 1)
            org = org.replace("'", "''")
            repo_name = repo_name.replace("'", "''")
            repo_conditions.append(f"(p.artifact_namespace = '{org}' AND p.artifact_name = '{repo_name}')")

    if not repo_conditions:
        return pd.DataFrame()

    repo_condition_str = " OR ".join(repo_conditions)
    results = []

    # Execute user-to-repo query
    try:
        user_to_repo_query = build_user_to_repo_query(users_str, repo_condition_str, date_filter, bot_filter)
        batch_df1 = client.to_pandas(user_to_repo_query)
        if not batch_df1.empty:
            results.append(batch_df1)
    except Exception as e:
        print(f"    Error in user-to-repo query: {e}")

    # Execute repo-to-user query
    try:
        repo_to_user_query = build_repo_to_user_query(users_str, repo_condition_str, date_filter, bot_filter)
        batch_df2 = client.to_pandas(repo_to_user_query)
        if not batch_df2.empty:
            results.append(batch_df2)
    except Exception as e:
        print(f"    Error in repo-to-user query: {e}")

    # Combine results
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()

def create_client():
    """Create a new OSO client for thread-safe operations."""
    api_key = os.getenv('OSO_API_KEY')
    if not api_key:
        raise ValueError("OSO API key required")

    import pyoso
    os.environ["OSO_API_KEY"] = api_key
    return pyoso.Client()

def process_batch_wrapper_with_date_filter(batch_info, date_filter, config=None):
    """Wrapper function for batch processing with date filter."""
    batch_num, batch_pairs, total_batches = batch_info

    # Create a new client for this thread
    try:
        client = create_client()
    except Exception as e:
        print(f"    ✗ Batch {batch_num}: Failed to create client: {e}")
        return batch_num, None, f"Failed to create client: {e}"

    print(f"  Processing batch {batch_num}/{total_batches} ({len(batch_pairs)} pairs)...")

    # Process batch with optimized query and comprehensive error handling
    try:
        batch_df = process_batch_optimized(batch_pairs, client, date_filter, config)

        if not batch_df.empty:
            # Round trust values
            batch_df['v'] = batch_df['v'].round(6)
            print(f"    ✓ Batch {batch_num}: Found {len(batch_df)} relationships")
            return batch_num, batch_df, None
        else:
            print(f"    ✓ Batch {batch_num}: No relationships found")
            return batch_num, None, None

    except Exception as e:
        import traceback
        error_msg = f"Batch processing failed: {str(e)}"
        print(f"    ✗ Batch {batch_num}: {error_msg}")
        return batch_num, None, error_msg

def generate_trust_relationships():
    """Generate trust relationships with simple job queue processing."""

    # Get API key
    api_key = os.getenv('OSO_API_KEY')
    if not api_key:
        print("ERROR: OSO API key required. Set OSO_API_KEY environment variable.")
        sys.exit(1)

    # Load config for days_back
    import tomli
    try:
        with open("config.toml", "rb") as f:
            config = tomli.load(f)
    except FileNotFoundError:
        print("Warning: config.toml not found, using default settings")
        config = {}

    # Build date filter
    date_filter = ""
    days_back = config.get("general", {}).get("days_back", 0)
    if days_back > 0:
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=days_back)
        date_filter = f"AND e.bucket_day >= DATE '{cutoff_date.strftime('%Y-%m-%d')}'"
        print(f"Using date filter: last {days_back} days (from {cutoff_date.strftime('%Y-%m-%d')})")
    else:
        print("Using all historical data (no date filter)")

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

    trust_dir = Path("trust")
    trust_dir.mkdir(exist_ok=True)

    # Load data efficiently
    user_repo_pairs, all_repos = load_user_repo_pairs()

    if not user_repo_pairs:
        print("ERROR: No user-repo pairs found in raw data!")
        return

    # Convert to list for batch processing
    pairs_list = list(user_repo_pairs)
    batch_size = 250
    total_batches = (len(pairs_list) + batch_size - 1) // batch_size

    print(f"Processing {len(pairs_list)} user-repo pairs in {total_batches} batches of {batch_size}...")
    print("Using sequential processing with retry queue...")

    # Initialize output
    output_file = trust_dir / "github.csv"
    all_results = []

    # Create job queue
    job_queue = []
    for batch_start in range(0, len(pairs_list), batch_size):
        batch_pairs = pairs_list[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        job_queue.append((batch_num, batch_pairs, total_batches))

    try:
        # Process jobs in queue sequentially
        while job_queue:
            # Take one job from queue
            current_job = job_queue.pop(0)
            batch_num, batch_pairs, total_batches = current_job

            # Process single job with date filter
            try:
                # Pass date_filter and config to process_batch_wrapper
                batch_num, batch_df, error = process_batch_wrapper_with_date_filter(current_job, date_filter, config)

                if error:
                    print(f"    ✗ Batch {batch_num} failed: {error}")
                    # Add failed job back to the end of the queue for retry
                    print(f"    Adding batch {batch_num} back to queue for retry")
                    job_queue.append(current_job)
                else:
                    if batch_df is not None:
                        all_results.append(batch_df)
            except Exception as e:
                error_msg = str(e)
                if "Expecting value" in error_msg:
                    print(f"    ✗ Batch {batch_num} failed: API returned empty response (likely rate limited)")
                else:
                    print(f"    ✗ Batch {batch_num} failed: {error_msg}")
                # Add failed job back to the end of the queue for retry
                print(f"    Adding batch {batch_num} back to queue for retry")
                job_queue.append(current_job)

        # Final save with complete aggregation
        if all_results:
            print(f"\n💾 Saving final results from {len(all_results)} successful batches...")
            final_df = pd.concat(all_results, ignore_index=True)
            final_df = final_df.groupby(['i', 'j'], as_index=False)['v'].sum()
            final_df['v'] = final_df['v'].round(6)

            # Filter out very small trust values to reduce noise
            final_df = final_df[final_df['v'] >= 0.1]

            final_df.to_csv(output_file, index=False, mode='w')
            print(f"✓ Final results: {len(final_df)} trust relationships saved to {output_file}")
        else:
            print("❌ No trust relationships found!")

    except Exception as e:
        print(f"ERROR: Processing failed: {e}")
        return


if __name__ == "__main__":
    try:
        generate_trust_relationships()
    except Exception as e:
        print(f"ERROR: Processing failed: {e}")
