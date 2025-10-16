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

from datetime import datetime, timedelta

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


def build_user_to_repo_query(users_str, repo_condition_str, date_filter="", bot_filter="", weights=None):
    """Build query to get raw user-to-repo events without applying weights."""
    return f"""
    SELECT
        u.artifact_name AS i,
        CONCAT(p.artifact_namespace, '/', p.artifact_name) AS j,
        e.event_type,
        SUM(e.amount) AS amount
    FROM int_events_daily__github AS e
    JOIN int_github_users AS u ON e.from_artifact_id = u.artifact_id
    JOIN artifacts_by_project_v1 AS p ON e.to_artifact_id = p.artifact_id
    WHERE u.artifact_name IN ('{users_str}')
      AND ({repo_condition_str})
      AND e.event_type IN ('COMMIT_CODE', 'PULL_REQUEST_OPENED', 'PULL_REQUEST_MERGED', 'STARRED', 'ISSUE_OPENED', 'FORKED')
      {bot_filter}
      AND p.artifact_source = 'GITHUB'
      {date_filter}
    GROUP BY u.artifact_name, CONCAT(p.artifact_namespace, '/', p.artifact_name), e.event_type
    HAVING SUM(e.amount) > 0
    ORDER BY u.artifact_name, CONCAT(p.artifact_namespace, '/', p.artifact_name), e.event_type
    """

def apply_weights_to_events(events_df, user_to_repo_weights, repo_to_user_weights, interaction_type):
    """Apply weights to raw event data after database retrieval."""
    if events_df.empty:
        return pd.DataFrame()

    # Default weights
    default_user_to_repo = {
        'COMMIT_CODE': 5,
        'PULL_REQUEST_OPENED': 20,
        'PULL_REQUEST_MERGED': 10,
        'STARRED': 5,
        'ISSUE_OPENED': 10,
        'FORKED': 1
    }

    default_repo_to_user = {
        'COMMIT_CODE': 3,
        'PULL_REQUEST_OPENED': 5,
        'PULL_REQUEST_MERGED': 1
    }

    # Merge with defaults and convert to uppercase
    if interaction_type == 'user_to_repo':
        weights = {**default_user_to_repo}
        if user_to_repo_weights:
            weights.update({k.upper(): v for k, v in user_to_repo_weights.items()})
    else:  # repo_to_user
        weights = {**default_repo_to_user}
        if repo_to_user_weights:
            weights.update({k.upper(): v for k, v in repo_to_user_weights.items()})

    # Apply weights to each event
    weighted_results = []
    for _, row in events_df.iterrows():
        event_type = row['event_type']
        weight = weights.get(event_type, 0)

        if weight > 0:
            weighted_results.append({
                'i': row['i'],
                'j': row['j'],
                'v': float(row['amount'] * weight)
            })

    # Convert to DataFrame and group by i,j to sum weights
    if weighted_results:
        result_df = pd.DataFrame(weighted_results)
        result_df = result_df.groupby(['i', 'j'], as_index=False)['v'].sum()
        return result_df
    else:
        return pd.DataFrame()

def extract_detailed_interactions_from_batch(batch_pairs, client, date_filter="", bot_filter=""):
    """Extract detailed interactions by querying raw event data for the batch pairs."""
    from datetime import datetime

    # Extract users and repos from batch
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
            org = org.replace("'", "''")
            repo_name = repo_name.replace("'", "''")
            repo_conditions.append(f"(p.artifact_namespace = '{org}' AND p.artifact_name = '{repo_name}')")

    if not repo_conditions:
        return []

    repo_condition_str = " OR ".join(repo_conditions)

    # Query for detailed event data
    detailed_query = f"""
    SELECT
        u.artifact_name AS user_name,
        CONCAT(p.artifact_namespace, '/', p.artifact_name) AS repo_name,
        e.event_type,
        SUM(e.amount) AS total_amount
    FROM int_events_daily__github AS e
    JOIN int_github_users AS u ON e.from_artifact_id = u.artifact_id
    JOIN artifacts_by_project_v1 AS p ON e.to_artifact_id = p.artifact_id
    WHERE u.artifact_name IN ('{users_str}')
      AND ({repo_condition_str})
      AND e.event_type IN ('COMMIT_CODE', 'PULL_REQUEST_OPENED', 'PULL_REQUEST_MERGED', 'STARRED', 'ISSUE_OPENED', 'FORKED')
      {bot_filter}
      AND p.artifact_source = 'GITHUB'
      {date_filter}
    GROUP BY u.artifact_name, CONCAT(p.artifact_namespace, '/', p.artifact_name), e.event_type
    HAVING SUM(e.amount) > 0
    """

    try:
        detailed_df = client.to_pandas(detailed_query)
        detailed_interactions = []

        if not detailed_df.empty:
            for _, row in detailed_df.iterrows():
                event_type = row['event_type']
                user = row['user_name']
                repo = row['repo_name']
                total_amount = int(row['total_amount'])

                # Store aggregated user_to_repo interaction
                interaction = {
                    "user": user,
                    "repo": repo,
                    "event_type": event_type,
                    "event_count": total_amount
                }
                detailed_interactions.append(interaction)

        return detailed_interactions
    except Exception as e:
        print(f"Warning: Could not extract detailed interactions: {e}")
        return []

def build_repo_to_user_query(users_str, repo_condition_str, date_filter="", bot_filter="", weights=None):
    """Build query to get raw repo-to-user events without applying weights."""
    return f"""
    SELECT
        CONCAT(p.artifact_namespace, '/', p.artifact_name) AS i,
        u.artifact_name AS j,
        e.event_type,
        SUM(e.amount) AS amount
    FROM int_events_daily__github AS e
    JOIN int_github_users AS u ON e.from_artifact_id = u.artifact_id
    JOIN artifacts_by_project_v1 AS p ON e.to_artifact_id = p.artifact_id
    WHERE u.artifact_name IN ('{users_str}')
      AND ({repo_condition_str})
      AND e.event_type IN ('COMMIT_CODE', 'PULL_REQUEST_OPENED', 'PULL_REQUEST_MERGED')
      {bot_filter}
      AND p.artifact_source = 'GITHUB'
      {date_filter}
    GROUP BY CONCAT(p.artifact_namespace, '/', p.artifact_name), u.artifact_name, e.event_type
    HAVING SUM(e.amount) > 0
    ORDER BY CONCAT(p.artifact_namespace, '/', p.artifact_name), u.artifact_name, e.event_type
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
        ("seed_repos.csv", "repository_name"),
        ("extended_repos.csv", "repository_name")
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
        ("seed_contributors.csv", "contributor_handle", "repository_name"),
        ("extended_contributors.csv", "contributor_handle", "repository_name")
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

def load_interactions_cache(days_back):
    """Load cached interactions from cache/interactions_{days_back}.csv"""
    cache_dir = Path("cache")
    cache_file = cache_dir / f"interactions_{days_back}.csv"

    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file)
            # Convert DataFrame to list of dictionaries for compatibility
            interactions = df.to_dict('records')
            return {"interactions": interactions, "metadata": {"days_back": days_back, "last_updated": None}}
        except Exception as e:
            print(f"Warning: Error loading interactions cache: {e}")

    return {"interactions": [], "metadata": {"days_back": days_back, "last_updated": None}}

def save_interactions_cache(interactions_data, days_back):
    """Save interactions to cache/interactions_{days_back}.csv"""
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"interactions_{days_back}.csv"

    try:
        # Convert interactions data to DataFrame with simplified format
        df_data = []
        for interaction in interactions_data:
            df_data.append({
                'user': interaction.get('user', ''),
                'repo': interaction.get('repo', ''),
                'event_type': interaction.get('event_type', ''),
                'event_count': interaction.get('event_count', interaction.get('value', 0))
            })

        df = pd.DataFrame(df_data)
        df.to_csv(cache_file, index=False)

        print(f"✓ Saved {len(interactions_data)} interactions to cache/{cache_file.name}")

    except Exception as e:
        print(f"Warning: Error saving interactions cache: {e}")

def append_interactions_to_cache(new_interactions, days_back):
    """Append new interactions to existing cache"""
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    try:
        # Load existing cache
        existing_cache = load_interactions_cache(days_back)
        existing_interactions = existing_cache.get("interactions", [])

        # Combine with new interactions
        all_interactions = existing_interactions + new_interactions

        # Save combined data
        save_interactions_cache(all_interactions, days_back)

    except Exception as e:
        print(f"Warning: Error appending to interactions cache: {e}")

def filter_cached_interactions_by_date(cached_interactions, current_days_back):
    """Filter cached interactions based on current days_back setting"""
    if current_days_back == 0:  # All time
        return cached_interactions

    cutoff_date = datetime.now() - timedelta(days=current_days_back)
    cutoff_iso = cutoff_date.isoformat()

    filtered = []
    for interaction in cached_interactions:
        if 'bucket_day' in interaction and interaction['bucket_day'] >= cutoff_iso[:10]:  # Compare date part
            filtered.append(interaction)

    return filtered

def convert_query_results_to_cache_format(df, interaction_type):
    """Convert DataFrame results to cache format"""
    cache_entries = []

    for _, row in df.iterrows():
        cache_entry = {
            "user": row.get('user_name', row.get('i', '')),
            "repo": row.get('repo_name', row.get('j', '')),
            "event_type": row.get('event_type', ''),
            "interaction_type": interaction_type,
            "value": int(row.get('amount', row.get('v', 0)))
        }
        cache_entries.append(cache_entry)

    return cache_entries

def load_processed_pairs_from_cache(days_back):
    """Load already processed pairs from CSV cache"""
    cache_dir = Path("cache")
    cache_file = cache_dir / f"interactions_{days_back}.csv"

    processed_pairs = set()

    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file)
            # Extract unique user-repo pairs from cached interactions
            for _, row in df.iterrows():
                user = row.get('user', '')
                repo = row.get('repo', '')
                if user and repo:
                    processed_pairs.add((user, repo))

            print(f"✓ Found {len(processed_pairs)} processed pairs in cache/{cache_file.name}")

        except Exception as e:
            print(f"Warning: Error loading processed pairs from cache: {e}")

    return processed_pairs

def get_unprocessed_pairs(user_repo_pairs, days_back):
    """Filter out already processed pairs and return only unprocessed ones"""
    print("Checking for already processed pairs...")
    print(f"  Looking for cache file with days_back >= {days_back}")

    # Find cache file with same or higher days_back value
    cache_dir = Path("cache")
    best_cache_days = None

    if cache_dir.exists():
        for cache_file in cache_dir.glob("interactions_*.csv"):
            try:
                cache_days = int(cache_file.stem.split('_')[1])
                if cache_days >= days_back:
                    if best_cache_days is None or cache_days < best_cache_days:
                        best_cache_days = cache_days
            except (ValueError, IndexError):
                continue

    if best_cache_days is not None:
        print(f"  ✓ Using cache: interactions_{best_cache_days}.csv")
        processed_pairs = load_processed_pairs_from_cache(best_cache_days)

        if processed_pairs:
            unprocessed_pairs = user_repo_pairs - processed_pairs
            print(f"  → {len(unprocessed_pairs)} pairs still need processing")
            print(f"  → {len(processed_pairs)} pairs already cached")
            return unprocessed_pairs, processed_pairs

    print("✓ No compatible cache found, processing all pairs")
    return user_repo_pairs, set()

def load_cached_interactions_for_trust(days_back, config=None):
    """Load cached interactions and convert to trust format using config weights"""
    cache_dir = Path("cache")
    cache_file = cache_dir / f"interactions_{days_back}.csv"

    cached_trust_data = []

    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file)
            print(f"✓ Loading {len(df)} cached interactions for trust calculation...")

            # Get weights from config or use defaults
            if config:
                user_to_repo_weights = config.get("weights", {}).get("user_to_repo", {})
                repo_to_user_weights = config.get("weights", {}).get("repo_to_user", {})
            else:
                user_to_repo_weights = {}
                repo_to_user_weights = {}

            # Default weights
            default_user_to_repo = {
                'commit_code': 5, 'pull_request_opened': 20, 'pull_request_merged': 10,
                'starred': 5, 'issue_opened': 10, 'forked': 1
            }
            default_repo_to_user = {
                'commit_code': 3, 'pull_request_opened': 5, 'pull_request_merged': 1
            }

            # Merge with defaults and convert to uppercase
            user_to_repo_weights = {**default_user_to_repo, **{k.lower(): v for k, v in user_to_repo_weights.items()}}
            repo_to_user_weights = {**default_repo_to_user, **{k.lower(): v for k, v in repo_to_user_weights.items()}}

            # Convert keys to uppercase for matching
            user_to_repo_weights = {k.upper(): v for k, v in user_to_repo_weights.items()}
            repo_to_user_weights = {k.upper(): v for k, v in repo_to_user_weights.items()}

            # Convert cached interactions to trust relationships using config weights
            for _, row in df.iterrows():
                user = row.get('user', '')
                repo = row.get('repo', '')
                event_type = row.get('event_type', '').upper()
                event_count = int(row.get('event_count', 0))

                if user and repo and event_count > 0:
                    # Apply user_to_repo weights
                    user_to_repo_weight = user_to_repo_weights.get(event_type, 0)
                    if user_to_repo_weight > 0:
                        cached_trust_data.append({
                            'i': user,
                            'j': repo,
                            'v': float(event_count * user_to_repo_weight)
                        })

                    # Apply repo_to_user weights for commit-type events
                    if event_type in ('COMMIT_CODE', 'PULL_REQUEST_OPENED', 'PULL_REQUEST_MERGED'):
                        repo_to_user_weight = repo_to_user_weights.get(event_type, 0)
                        if repo_to_user_weight > 0:
                            cached_trust_data.append({
                                'i': repo,
                                'j': user,
                                'v': float(event_count * repo_to_user_weight)
                            })

            if cached_trust_data:
                cached_df = pd.DataFrame(cached_trust_data)
                # Group by i,j and sum values
                cached_df = cached_df.groupby(['i', 'j'], as_index=False)['v'].sum()
                print(f"✓ Converted to {len(cached_df)} cached trust relationships using config weights")
                return cached_df

        except Exception as e:
            print(f"Warning: Error loading cached interactions: {e}")

    return pd.DataFrame()

def convert_cached_interactions_to_trust(cached_interactions, config):
    """Convert cached interactions to trust relationship format"""
    if not cached_interactions:
        return pd.DataFrame()

    # Get weights from config
    user_to_repo_weights = config.get("weights", {}).get("user_to_repo", {})
    repo_to_user_weights = config.get("weights", {}).get("repo_to_user", {})

    # Default weights
    default_user_to_repo = {'commit_code': 5, 'pull_request_opened': 20, 'pull_request_merged': 10, 'starred': 5, 'issue_opened': 10, 'forked': 1}
    default_repo_to_user = {'commit_code': 3, 'pull_request_opened': 5, 'pull_request_merged': 1}

    # Merge with defaults
    user_to_repo_weights = {**default_user_to_repo, **{k.upper(): v for k, v in user_to_repo_weights.items()}}
    repo_to_user_weights = {**default_repo_to_user, **{k.upper(): v for k, v in repo_to_user_weights.items()}}

    trust_data = []

    for interaction in cached_interactions:
        event_type = interaction.get('event_type', '').upper()
        interaction_type = interaction.get('interaction_type', '')
        amount = interaction.get('value', interaction.get('amount', 0))
        user = interaction.get('user', '')
        repo = interaction.get('repo', '')

        # Calculate weighted value
        if interaction_type == 'user_to_repo':
            weight = user_to_repo_weights.get(event_type, 0)
            i, j = user, repo
        elif interaction_type == 'repo_to_user':
            weight = repo_to_user_weights.get(event_type, 0)
            i, j = repo, user
        else:
            continue

        if weight > 0:
            trust_data.append({
                'i': i,
                'j': j,
                'v': amount * weight
            })

    if trust_data:
        df = pd.DataFrame(trust_data)
        # Group by i,j and sum values
        df = df.groupby(['i', 'j'], as_index=False)['v'].sum()
        return df

    return pd.DataFrame()

def process_batch_optimized(batch_pairs, client, date_filter="", config=None, use_detailed_cache=False):
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

    users_str = "', '".join(sorted(users_in_batch))

    # Build repo conditions more efficiently
    repo_conditions = []
    for repo in repos_in_batch:
        if '/' in repo:
            org, repo_name = repo.split('/', 1)
            org = org.replace("'", "''")
            repo_name = repo_name.replace("'", "''")
            repo_conditions.append(f"(p.artifact_namespace = '{org}' AND p.artifact_name = '{repo_name}')")

    if not repo_conditions:
        return pd.DataFrame(), []

    repo_condition_str = " OR ".join(sorted(repo_conditions))
    results = []
    detailed_interactions = []

    # Create detailed interactions for caching with individual event types
    if use_detailed_cache:
        # Extract detailed interactions from batch pairs
        detailed_interactions = extract_detailed_interactions_from_batch(
            batch_pairs, client, date_filter, bot_filter
        )

    # Get weights from config
    user_to_repo_weights = config.get("weights", {}).get("user_to_repo", {})
    repo_to_user_weights = config.get("weights", {}).get("repo_to_user", {})

    # Execute user-to-repo query
    user_to_repo_query = build_user_to_repo_query(users_str, repo_condition_str, date_filter, bot_filter)
    raw_user_to_repo_df = client.to_pandas(user_to_repo_query)
    if not raw_user_to_repo_df.empty:
        # Apply weights after data retrieval
        weighted_user_to_repo_df = apply_weights_to_events(raw_user_to_repo_df, user_to_repo_weights, repo_to_user_weights, 'user_to_repo')
        if not weighted_user_to_repo_df.empty:
            results.append(weighted_user_to_repo_df)

    # Execute repo-to-user query
    repo_to_user_query = build_repo_to_user_query(users_str, repo_condition_str, date_filter, bot_filter)
    raw_repo_to_user_df = client.to_pandas(repo_to_user_query)
    if not raw_repo_to_user_df.empty:
        # Apply weights after data retrieval
        weighted_repo_to_user_df = apply_weights_to_events(raw_repo_to_user_df, user_to_repo_weights, repo_to_user_weights, 'repo_to_user')
        if not weighted_repo_to_user_df.empty:
            results.append(weighted_repo_to_user_df)

    # Combine results
    combined_results = pd.DataFrame()
    if results:
        combined_results = pd.concat(results, ignore_index=True)



    return combined_results, detailed_interactions

def create_client():
    """Create a new OSO client for thread-safe operations."""
    api_key = os.getenv('OSO_API_KEY')
    if not api_key:
        raise ValueError("OSO API key required")

    import pyoso
    os.environ["OSO_API_KEY"] = api_key
    return pyoso.Client()

def process_batch_wrapper_with_date_filter(batch_info, date_filter, config=None, use_detailed_cache=False, all_interactions=None):
    """Wrapper function for batch processing with date filter."""
    batch_num, batch_pairs, total_batches = batch_info

    # Create a new client for this thread
    try:
        client = create_client()
    except Exception as e:
        print(f"    ✗ Batch {batch_num}: Failed to create client: {e}")
        return batch_num, None, None, f"Failed to create client: {e}"

    print(f"  Processing batch {batch_num}/{total_batches} ({len(batch_pairs)} pairs)...")

    # Process batch with optimized query and comprehensive error handling
    try:
        batch_df, batch_interactions = process_batch_optimized(batch_pairs, client, date_filter, config, use_detailed_cache)

        if use_detailed_cache and all_interactions is not None:
            all_interactions.extend(batch_interactions)

        if not batch_df.empty:
            # Round trust values
            batch_df['v'] = batch_df['v'].round(6)
            print(f"    ✓ Batch {batch_num}: Found {len(batch_df)} relationships")
            return batch_num, batch_df, batch_interactions, None
        else:
            print(f"    ✓ Batch {batch_num}: No relationships found")
            return batch_num, None, batch_interactions, None

    except Exception as e:
        import traceback
        error_msg = f"Batch processing failed: {str(e)}"
        print(f"    ✗ Batch {batch_num}: {error_msg}")
        return batch_num, None, [], error_msg

def generate_trust_relationships(config_path="config.toml"):
    """Generate trust relationships with simple job queue processing."""

    # Get API key
    api_key = os.getenv('OSO_API_KEY')
    if not api_key:
        print("ERROR: OSO API key required. Set OSO_API_KEY environment variable.")
        sys.exit(1)

    # Load config for days_back
    import tomli
    try:
        with open(config_path, "rb") as f:
            config = tomli.load(f)
        print(f"✓ Loaded configuration from {config_path}")
    except FileNotFoundError:
        print(f"Warning: {config_path} not found, using default settings")
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

    # Get unprocessed pairs by filtering out already processed ones
    pairs_to_process, processed_pairs = get_unprocessed_pairs(user_repo_pairs, days_back)

    # Load cached trust relationships for already processed pairs
    cached_trust_relationships = []
    cached_trust_df = load_cached_interactions_for_trust(days_back, config)
    if not cached_trust_df.empty:
        cached_trust_relationships.append(cached_trust_df)

    # Process remaining pairs if needed
    new_results = []
    new_interactions = []

    if pairs_to_process:
        # Convert to sorted list for deterministic batch processing
        pairs_list = sorted(list(pairs_to_process))
        batch_size = 250
        total_batches = (len(pairs_list) + batch_size - 1) // batch_size

        print(f"Processing {len(pairs_list)} user-repo pairs in {total_batches} batches of {batch_size}...")
        print("Using sequential processing with retry queue...")

        # Create job queue
        job_queue = []
        for batch_start in range(0, len(pairs_list), batch_size):
            batch_pairs = pairs_list[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            job_queue.append((batch_num, batch_pairs, total_batches))

        try:
            # Process jobs in queue sequentially
            completed_batches = 0
            while job_queue:
                # Take one job from queue
                current_job = job_queue.pop(0)
                batch_num, batch_pairs, total_batches = current_job

                # Process single job with date filter
                try:
                    # Pass date_filter and config to process_batch_wrapper
                    batch_num, batch_df, batch_interactions, error = process_batch_wrapper_with_date_filter(
                        current_job, date_filter, config, use_detailed_cache=True
                    )

                    if error:
                        print(f"    ✗ Batch {batch_num} failed: {error}")
                        # Add failed job back to the end of the queue for retry
                        print(f"    Adding batch {batch_num} back to queue for retry")
                        job_queue.append(current_job)
                    else:
                        if batch_df is not None:
                            new_results.append(batch_df)
                        if batch_interactions:
                            new_interactions.extend(batch_interactions)

                        completed_batches += 1

                        # Save to cache every 10 iterations
                        if completed_batches % 10 == 0 and new_interactions:
                            print(f"    💾 Saving {len(new_interactions)} interactions to cache (checkpoint at batch {completed_batches})")
                            append_interactions_to_cache(new_interactions, days_back)
                            new_interactions.clear()  # Clear the list to avoid duplicates

                except Exception as e:
                    error_msg = str(e)
                    if "Expecting value" in error_msg:
                        print(f"    ✗ Batch {batch_num} failed: API returned empty response (likely rate limited)")
                    else:
                        print(f"    ✗ Batch {batch_num} failed: {error_msg}")
                    # Add failed job back to the end of the queue for retry
                    print(f"    Adding batch {batch_num} back to queue for retry")
                    job_queue.append(current_job)

        except Exception as e:
            print(f"ERROR: Processing failed: {e}")
            return

    # Combine cached and new results
    all_results = cached_trust_relationships + new_results

    # Save any remaining interactions to cache
    if new_interactions:
        print(f"💾 Saving final {len(new_interactions)} interactions to cache")
        append_interactions_to_cache(new_interactions, days_back)

    # Initialize output
    output_file = trust_dir / "github.csv"

    # Final save with complete aggregation
    if all_results:
        print(f"\n💾 Saving final results from {len(all_results)} result sets...")
        final_df = pd.concat(all_results, ignore_index=True)
        final_df = final_df.groupby(['i', 'j'], as_index=False)['v'].sum()
        final_df['v'] = final_df['v'].round(6)

        # Filter out very small trust values to reduce noise
        final_df = final_df[final_df['v'] >= 0.1]

        final_df.to_csv(output_file, index=False, mode='w')
        print(f"✓ Final results: {len(final_df)} trust relationships saved to {output_file}")
    else:
        print("❌ No trust relationships found!")


if __name__ == "__main__":
    import sys
    try:
        # Check for custom config file argument
        config_file = sys.argv[1] if len(sys.argv) > 1 else "config.toml"
        generate_trust_relationships(config_file)
    except KeyboardInterrupt:
        print("\nCancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Processing failed: {e}")
        sys.exit(1)
