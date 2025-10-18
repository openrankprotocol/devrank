#!/usr/bin/env python3
"""
Crypto Ecosystem Contributor Network Analyzer - Repository-Based Pipeline

This script orchestrates a comprehensive crypto ecosystem analysis using a
repository-based approach:

1. Load seed repositories from config.toml
2. Find core contributors for seed repositories
3. Find extended repositories that core contributors contributed to
4. Find extended contributors from the extended repositories

Prerequisites:
1. Create account at www.opensource.observer
2. Generate API key in Account Settings > API Keys
3. Set OSO_API_KEY environment variable or in .env file
4. Install dependencies: pip install pyoso pandas python-dotenv tomli
5. Configure parameters in config.toml (use seed_repos instead of seed_orgs)

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
import time
import json
from collections import deque
# Removed threading imports

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
        "general", "analysis"
    ]

    for section in required_sections:
        if section not in config:
            print(f"ERROR: Missing required section [{section}] in config.toml")
            sys.exit(1)

    if not config["general"]["seed_repos"]:
        print("ERROR: No seed repositories specified in config.toml")
        sys.exit(1)

    return config

def get_interactions_cache_filename(days_back):
    """Generate interactions cache filename based on days_back."""
    return f"interactions_{days_back}.csv"


def load_cache(config):
    """Load all cache mappings derived from interactions cache."""
    days_back = config.get("general", {}).get("days_back", 0)

    cache_dir = Path("cache")
    cache_file = cache_dir / get_interactions_cache_filename(days_back)

    if cache_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(cache_file)

            # Build all mappings and event counts
            repo_contributors = {}
            contributor_repos = {}
            repo_event_counts = {}

            for _, row in df.iterrows():
                repo = row.get('repo', '')
                user = row.get('user', '')
                event_type = row.get('event_type', '').upper()
                event_count = int(row.get('event_count', 0))

                if repo and user:
                    # Build repo -> contributors mapping
                    if repo not in repo_contributors:
                        repo_contributors[repo] = []
                    if user not in repo_contributors[repo]:
                        repo_contributors[repo].append(user)

                    # Build contributor -> repos mapping
                    if user not in contributor_repos:
                        contributor_repos[user] = []
                    if repo not in contributor_repos[user]:
                        contributor_repos[user].append(repo)

                # Track commit counts
                if repo and event_type == 'COMMIT_CODE':
                    repo_event_counts[repo] = repo_event_counts.get(repo, 0) + event_count

            print(f"✓ Loaded cache from interactions: {cache_file.name}")
            print(f"  Cached repositories: {len(repo_contributors):,}")
            print(f"  Cached contributors: {len(contributor_repos):,}")

            return {
                'repo_contributors': repo_contributors,
                'contributor_repos': contributor_repos,
                'event_count': repo_event_counts
            }
        except Exception as e:
            print(f"Warning: Could not load interactions cache file {cache_file}: {e}")
            return {'repo_contributors': {}, 'contributor_repos': {}, 'event_count': {}}
    else:
        print(f"  Interactions cache file does not exist: {cache_file}")

    return {'repo_contributors': {}, 'contributor_repos': {}, 'event_count': {}}

def discover_seed_repositories(config):
    """
    Step 1: Load seed repositories from configuration.

    Returns:
        list: List of repository identifiers in "org/repo" format
    """

    print("🔍 Step 1: Discovering seed repositories...")
    # Get seed repositories from config
    seed_repos = config["general"]["seed_repos"]

    print(f"Loading {len(seed_repos)} seed repositories from configuration...")

    # Validate repositories exist
    api_key = config.get("api_key") or os.getenv("OSO_API_KEY")
    if api_key:
        try:
            from pyoso import Client
            client = Client(api_key=api_key)

            # Build conditions for all repos in one query
            repo_conditions = []
            for repo in seed_repos:
                if '/' in repo:
                    org, name = repo.split('/', 1)
                    repo_conditions.append(f"(artifact_namespace = '{org}' AND artifact_name = '{name}')")

            if repo_conditions:
                query = f"""
                SELECT artifact_namespace, artifact_name
                FROM artifacts_v1
                WHERE artifact_source = 'GITHUB'
                  AND ({' OR '.join(repo_conditions)})
                """
                result = client.to_pandas(query)
                found_repos = set(result['artifact_namespace'] + '/' + result['artifact_name'])

                for repo in seed_repos:
                    if repo not in found_repos:
                        print(f"  ⚠️ {repo} - not found in OSO database")
        except Exception:
            pass

    for repo in seed_repos:
        print(f"  ✓ {repo}")

    print(f"\n✓ Total seed repositories loaded: {len(seed_repos)}")

    if not seed_repos:
        print("ERROR: No seed repositories found in configuration!")
        sys.exit(1)

    # Save seed repositories list
    output_dir = Path(config.get("general", {}).get("output_dir", "./raw"))
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_repos_df = pd.DataFrame({
        'repository_name': seed_repos,
    })

    seed_file = output_dir / "seed_repos.csv"
    seed_repos_df.to_csv(seed_file, index=False)
    print(f"✓ Saved seed repositories: {seed_file}")

    return seed_repos


def find_core_contributors(seed_repos, config):
    """
    Step 2: Find core contributors for seed repositories using repo_to_contributors.py

    Returns:
        list: List of core contributor handles
    """

    print("👥 Step 2: Finding core contributors for seed repositories...")
    print(f"Finding contributors for {len(seed_repos)} seed repositories...")

    # Load existing cache
    cache = load_cache(config)

    # Filter repos that need processing (not in cache)
    repos_to_process = [repo for repo in seed_repos if repo not in cache['repo_contributors']]
    contributors_cached = []
    for repo in seed_repos:
        if repo in cache['repo_contributors']:
            for contributor in cache['repo_contributors'][repo]:
                contributors_cached.append({
                    'repository_name': repo,
                    'contributor_handle': contributor
                })

    if repos_to_process:
        print(f"  Processing {len(repos_to_process)} uncached seed repositories...")

        # Configure parameters from config.toml
        min_commits = config.get("analysis", {}).get("min_commits", 10)
        date_filter_days = config.get("general", {}).get("days_back", 0)

        # Use repo_to_contributors module for uncached repos
        contributors_df = repo_to_contributors.get_contributors_by_repos(
            repo_identifiers=repos_to_process,
            min_commits=min_commits,
            date_filter_days=date_filter_days,
            config=config
        )

        if not contributors_df.empty:
            # Save contributors data
            output_dir = Path(config.get("general", {}).get("output_dir", "./raw"))
            output_dir.mkdir(parents=True, exist_ok=True)
            core_contrib_file = output_dir / "seed_contributors.csv"
            # add contributors_cached to contributors_df
            if contributors_cached:
                cached_df = pd.DataFrame(contributors_cached)
                contributors_df = pd.concat([contributors_df, cached_df], ignore_index=True).drop_duplicates(subset=['repository_name', 'contributor_handle'])
            contributors_df.to_csv(core_contrib_file, index=False)
            print(f"✓ Saved seed contributors: {core_contrib_file}")

            # Get unique contributor handles
            core_contributor_handles = contributors_df['contributor_handle'].unique().tolist()
            print(f"✓ Found {len(core_contributor_handles)} core contributors")

            return core_contributor_handles
        else:
            return list(set([contrib['contributor_handle'] for contrib in contributors_cached]))

    else:
        print(f"  All {len(seed_repos)} seed repositories found in cache")

def find_extended_repos_by_stars(core_contributors, config):
    """
    Find extended repos that core contributors contributed to, sorted by stars
    Uses chunking approach to handle large contributor lists.
    """
    if not core_contributors:
        print("ERROR: No core contributors provided")
        return []

    # Get API key
    api_key = config.get("api_key") or os.getenv("OSO_API_KEY")
    if not api_key:
        print("ERROR: OSO_API_KEY is required")
        return []

    try:
        from pyoso import Client
        client = Client(api_key=api_key)
        print("✓ OSO client initialized")
    except ImportError:
        print("ERROR: pyoso library not found. Install with: pip install pyoso")
        return []
    except Exception as e:
        print(f"ERROR: Failed to initialize OSO client: {e}")
        return []

    min_core_contributors = config.get("analysis", {}).get("min_core_contributors", 2)
    min_commits = config.get("analysis", {}).get("min_commits", 10)
    max_extended_repos = config.get("analysis", {}).get("max_extended_repos", 200)

    # Load cache to filter out contributors we've already processed
    cache = load_cache(config)

    # Filter out contributors that are already in the cache
    uncached_contributors = [c for c in core_contributors if c not in cache['contributor_repos']]

    print(f"Finding extended repos from {len(uncached_contributors)} uncached core contributors...")
    print(f"  Filters: min_commits={min_commits}, min_core_contributors={min_core_contributors}")
    print(f"  Will select top {max_extended_repos} repos by star count")

    if not uncached_contributors:
        print("  All contributors are already cached - loading extended repos from cache")

        # Go through all core_contributors, find repos that they contributed to
        all_extended_repos = set()
        for contributor in core_contributors:
            if contributor in cache['contributor_repos']:
                repos = cache['contributor_repos'][contributor]
                all_extended_repos.update(repos)

        # Remove seed repos from extended repos
        seed_repos = config.get("general", {}).get("seed_repos", [])
        extended_repos_list = [repo for repo in all_extended_repos if repo not in seed_repos]

        # Sort by commit count and take top max_extended_repos
        extended_repos_sorted = sorted(extended_repos_list, key=lambda x: cache['event_count'].get(x, 0), reverse=True)
        extended_repos = extended_repos_sorted[:max_extended_repos]

        print(f"✓ Found {len(extended_repos)} extended repositories from cache")
        return extended_repos

    # Use filtered contributors for processing
    core_contributors = uncached_contributors

    # Process in chunks to avoid huge queries
    chunk_size = 100
    all_repo_stats = {}

    # Create chunks and queue
    chunk_queue = deque()
    for i in range(0, len(core_contributors), chunk_size):
        chunk = core_contributors[i:i+chunk_size]
        chunk_id = i//chunk_size + 1
        chunk_queue.append((chunk_id, chunk))

    total_chunks = len(chunk_queue)
    completed_chunks = 0

    print(f"  Processing {total_chunks} chunks with queue system...")

    # Process chunks with simple queue
    while chunk_queue:
        chunk_id, chunk = chunk_queue.popleft()
        print(f"  Processing chunk {chunk_id} ({len(chunk)} contributors)...")

        try:
            chunk_results = _process_contributor_chunk(chunk, config, client, min_commits)

            # Merge results
            for repo_name, stats in chunk_results.items():
                if repo_name in all_repo_stats:
                    all_repo_stats[repo_name]['contributors'].update(stats['contributors'])
                    all_repo_stats[repo_name]['total_commits'] += stats['total_commits']
                else:
                    all_repo_stats[repo_name] = stats

            completed_chunks += 1
            progress = (completed_chunks / total_chunks) * 100
            print(f"    ✓ Chunk {chunk_id} completed ({completed_chunks}/{total_chunks}, {progress:.1f}%)")

        except Exception as e:
            error_msg = str(e) if str(e) else f"{type(e).__name__}: Unable to parse error message"
            print(f"    ✗ Chunk {chunk_id} failed: {error_msg}")
            print(f"    → Adding back to queue")
            chunk_queue.append((chunk_id, chunk))

    print(f"  ✓ All chunks completed! Found repositories: {len(all_repo_stats)}")

    if not all_repo_stats:
        print("✗ No extended repositories found")
        return []

    # Filter by minimum core contributors
    filtered_repos = {}
    for repo_name, stats in all_repo_stats.items():
        contributor_count = len(stats['contributors'])
        if contributor_count >= min_core_contributors:
            filtered_repos[repo_name] = {
                'contributor_count': contributor_count,
                'total_commits': stats['total_commits']
            }

    if not filtered_repos:
        print(f"✗ No repositories meet minimum {min_core_contributors} core contributors")
        return []

    print(f"  Found {len(filtered_repos)} repositories meeting criteria")

    # Get top repos by commits (simpler than star ranking)
    sorted_repos = sorted(
        filtered_repos.items(),
        key=lambda x: x[1]['total_commits'],
        reverse=True
    )
    final_repos = [name for name, _ in sorted_repos[:max_extended_repos]]

    print(f"✓ Found {len(final_repos)} extended repositories")

    # Save results
    try:
        output_dir = Path(config.get("general", {}).get("output_dir", "./raw"))
        output_dir.mkdir(parents=True, exist_ok=True)

        extended_repos_df = pd.DataFrame({'repository_name': final_repos})
        extended_file = output_dir / "extended_repos.csv"
        extended_repos_df.to_csv(extended_file, index=False)
        print(f"✓ Saved extended repositories: {extended_file}")
    except Exception as e:
        print(f"Warning: Could not save results: {e}")

    return final_repos


def _process_contributor_chunk(contributors_chunk, config, client, min_commits):
    """Process a chunk of contributors and return repository statistics."""
    try:
        contributors_str = "', '".join([c.replace("'", "''") for c in contributors_chunk])

        # Build date filter
        date_filter = ""
        date_filter_days = config.get("general", {}).get("days_back", 0)
        if date_filter_days > 0:
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=date_filter_days)
            date_filter = f"AND e.bucket_day >= DATE '{cutoff_date.strftime('%Y-%m-%d')}'"

        # Get seed repos to exclude
        seed_repos = config.get("general", {}).get("seed_repos", [])
        seed_exclusion = ""
        if seed_repos:
            seed_conditions = []
            for repo in seed_repos:
                if '/' in repo:
                    org, name = repo.split('/', 1)
                    seed_conditions.append(f"(p.artifact_namespace = '{org}' AND p.artifact_name = '{name}')")
            if seed_conditions:
                seed_exclusion = f"AND NOT ({' OR '.join(seed_conditions)})"

        # Build bot filter
        filter_conditions = build_filter_conditions(config)
        bot_filter = f"AND {filter_conditions['bot_filter']}" if filter_conditions.get('bot_filter') else ""

        # Ensure min_commits is at least 1 to avoid huge result sets
        effective_min_commits = max(min_commits, 1)

        # Simple query to get repo commits by contributor
        query = f"""
        SELECT
            CONCAT(p.artifact_namespace, '/', p.artifact_name) as repository_name,
            u.artifact_name as contributor,
            SUM(e.amount) as commits
        FROM int_events_daily__github AS e
        JOIN int_github_users AS u ON e.from_artifact_id = u.artifact_id
        JOIN artifacts_v1 AS p ON e.to_artifact_id = p.artifact_id
        WHERE u.artifact_name IN ('{contributors_str}')
          AND e.event_type = 'COMMIT_CODE'
          AND p.artifact_source = 'GITHUB'
          {bot_filter}
          {date_filter}
          {seed_exclusion}
        GROUP BY p.artifact_namespace, p.artifact_name, u.artifact_name
        HAVING SUM(e.amount) >= {effective_min_commits}
        """

        print(f"    Query with {len(contributors_chunk)} contributors, min_commits={effective_min_commits}")

        df = client.to_pandas(query)
        if df.empty:
            print(f"    No repositories found for this chunk")
            return {}

        print(f"    Found {len(df)} repo-contributor pairs")

        # Aggregate by repository
        repo_stats = {}
        for _, row in df.iterrows():
            repo = row['repository_name']
            contributor = row['contributor']
            commits = row['commits']

            if repo not in repo_stats:
                repo_stats[repo] = {'contributors': set(), 'total_commits': 0}

            repo_stats[repo]['contributors'].add(contributor)
            repo_stats[repo]['total_commits'] += commits

        print(f"    Processed into {len(repo_stats)} unique repositories")
        return repo_stats

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else "Unable to parse error message"

        # Provide specific error context
        print(f"    ERROR in chunk processing ({error_type}): {error_msg}")
        if "JSON" in error_msg or "Expecting value" in error_msg:
            print(f"    → Likely API timeout or empty response")
        elif "timeout" in error_msg.lower():
            print(f"    → Query timeout - chunk may be too large")
        elif "rate" in error_msg.lower():
            print(f"    → Rate limiting - will retry with delay")

        raise  # Re-raise the exception for retry handling


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

    try:
        # Load existing cache
        cache = load_cache(config)

        # Filter repos that need processing (not in cache)
        repos_to_process = [repo for repo in extended_repos if repo not in cache['repo_contributors']]
        contributors_cached = []
        for repo in extended_repos:
            if repo in cache['repo_contributors']:
                for contributor in cache['repo_contributors'][repo]:
                    contributors_cached.append({
                        'repository_name': repo,
                        'contributor_handle': contributor
                    })

        if repos_to_process:
            print(f"  Processing {len(repos_to_process)} uncached repositories...")

            # Configure parameters from config.toml
            min_commits = config.get("analysis", {}).get("min_commits", 10)
            date_filter_days = config.get("general", {}).get("days_back", 0)

            # Use repo_to_contributors module for uncached repos
            extended_contributors_df = repo_to_contributors.get_contributors_by_repos(
                repo_identifiers=repos_to_process,
                min_commits=min_commits,
                date_filter_days=date_filter_days,
                config=config
            )

            if not extended_contributors_df.empty:
                # Save extended contributors data
                output_dir = Path(config.get("general", {}).get("output_dir", "./raw"))
                extended_contrib_file = output_dir / "extended_contributors.csv"
                # add contributors_cached to contributors_df
                if contributors_cached:
                    cached_df = pd.DataFrame(contributors_cached)
                    extended_contributors_df = pd.concat(
                        [extended_contributors_df, cached_df],
                        ignore_index=True
                    ).drop_duplicates(subset=['repository_name', 'contributor_handle'])
                extended_contributors_df.to_csv(extended_contrib_file, index=False)
                print(f"✓ Saved extended contributors: {extended_contrib_file}")
            else:
                print("No extended contributors found")

        else:
            print(f"  All {len(extended_repos)} repositories found in cache")

    except Exception as e:
        error_msg = str(e) if str(e) else f"{type(e).__name__}: Unable to parse error message"
        print(f"ERROR: Failed to find extended contributors: {error_msg}")
        return pd.DataFrame()

def main(config_path="config.toml"):
    """Main orchestration function."""

    # Load and validate configuration
    config = load_config(config_path)
    config = validate_config(config)
    date_filter_days = config.get('general', {}).get('days_back', 0)
    date_filter_text = 'All time' if date_filter_days == 0 else f'{date_filter_days} days back'

    try:
        output_dir = Path(config.get("general", {}).get("output_dir", "./raw"))

        # Step 1: Discover seed repositories from configuration
        seed_repos = discover_seed_repositories(config)

        # Step 2: Find core contributors for seed repositories
        core_contributors = find_core_contributors(seed_repos, config)

        # Check if extended analysis is enabled
        extended_analysis_enabled = config.get("analysis", {}).get("extended_analysis", True)

        if extended_analysis_enabled:
            print("📊 Extended analysis is enabled")

            # Step 3: Find extended repositories that core contributors contributed to
            print("🔍 Step 3: Finding extended repositories...")
            extended_repos = find_extended_repos_by_stars(core_contributors, config)

            # Step 4: Find extended contributors from extended repositories
            if extended_repos:
                print("👥 Step 4: Finding extended contributors...")
                find_extended_contributors(extended_repos, config)
            else:
                print("⚠️  No extended repositories found - skipping extended contributors")
        else:
            print("⚠️  Extended analysis is disabled - skipping extended repositories and contributors")
            print("   Only seed repositories and their core contributors will be analyzed")

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
