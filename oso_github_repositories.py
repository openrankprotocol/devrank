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


def retry_extended_analysis(func, *args, **kwargs):
    """
    Simple retry logic for extended analysis operations.
    Tries up to 3 times with fixed delays.
    """
    for attempt in range(3):
        try:
            if attempt > 0:
                print(f"  Retry attempt {attempt + 1}/3...")
            result = func(*args, **kwargs)
            if attempt > 0:
                print(f"✓ Success on retry attempt {attempt + 1}")
            return result

        except Exception as e:
            error_msg = str(e) if str(e) else f"{type(e).__name__}: Unable to parse error message"

            if attempt < 2:  # Will retry
                print(f"✗ Attempt failed: {error_msg}")
                print(f"  Retrying in 10 seconds...")
                time.sleep(10)
            else:  # Final failure
                print(f"✗ Final attempt failed: {error_msg}")
                print(f"  Giving up after 3 attempts")
                return [] if 'repos' in func.__name__ else pd.DataFrame()


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


def discover_seed_repositories(config):
    """
    Step 1: Load seed repositories from configuration.

    Returns:
        list: List of repository identifiers in "org/repo" format
    """

    # Check if seed repositories file already exists
    output_dir = Path(config.get("general", {}).get("output_dir", "./raw"))
    seed_file = output_dir / "seed_repos.csv"

    if seed_file.exists():
        print("✓ Seed repositories file already exists, loading from file...")
        seed_repos_df = pd.read_csv(seed_file)
        seed_repos = seed_repos_df['repository_name'].tolist()
        print(f"✓ Loaded {len(seed_repos)} seed repositories from {seed_file}")
        return seed_repos

    # Get seed repositories from config
    seed_repos = config["general"]["seed_repos"]

    print(f"Loading {len(seed_repos)} seed repositories from configuration...")

    # Validate repositories exist (optional check)
    # We'll skip API validation for now to keep it simple

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

    # Check if core contributors file already exists
    output_dir = Path(config.get("general", {}).get("output_dir", "./raw"))
    core_contrib_file = output_dir / "seed_contributors.csv"

    if core_contrib_file.exists():
        print("✓ Core contributors file already exists, loading from file...")
        core_contrib_df = pd.read_csv(core_contrib_file)
        core_contributors = core_contrib_df['contributor_handle'].unique().tolist()
        print(f"✓ Loaded {len(core_contributors)} core contributors from {core_contrib_file}")
        return core_contributors

    print(f"Finding contributors for {len(seed_repos)} seed repositories...")

    # Configure parameters from config.toml
    min_commits = config.get("analysis", {}).get("min_commits", 10)
    date_filter_days = config.get("general", {}).get("days_back", 0)

    # Use repo_to_contributors module
    contributors_df = repo_to_contributors.get_contributors_by_repos(
        repo_identifiers=seed_repos,
        min_commits=min_commits,
        date_filter_days=date_filter_days,
        config=config
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

    # Get all core contributors (no artificial limit)
    core_contributor_handles = contributor_summary.index.tolist()

    return core_contributor_handles


def find_extended_ecosystem(core_contributors, config):
    """
    Step 3: Find extended organizations and repositories using contributors_to_repos.py

    Returns:
        tuple: (extended_repos_list, extended_organizations_list)
    """

    print(f"Finding repositories for {len(core_contributors)} core contributors...")

    # Configure parameters from config.toml
    min_commits = config.get("analysis", {}).get("min_commits", 10)
    include_org_repos = True  # Always include organization repositories
    date_filter_days = config.get("general", {}).get("days_back", 0)

    # Use contributors_to_repos module
    repos_data = contributors_to_repos.get_repos_by_contributors(
        contributor_identifiers=core_contributors,
        min_commits=min_commits,
        include_org_repos=include_org_repos,
        date_filter_days=date_filter_days,
        config=config
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
        extended_repos.extend(org_repo_names)

        # Get unique organizations
        extended_orgs = organization_repos_df['organization'].unique().tolist()
        extended_organizations.extend(extended_orgs)

    # Remove duplicates and filter out seed repos if needed
    extended_repos = list(set(extended_repos))
    extended_organizations = list(set(extended_organizations))

    return extended_repos, extended_organizations


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

    print(f"Finding extended repos from {len(core_contributors)} core contributors...")
    print(f"  Filters: min_commits={min_commits}, min_core_contributors={min_core_contributors}")
    print(f"  Will select top {max_extended_repos} repos by star count")

    # Process in chunks to avoid huge queries
    chunk_size = 100  # Process 100 contributors at a time
    all_repo_stats = {}

    for i in range(0, len(core_contributors), chunk_size):
        chunk = core_contributors[i:i+chunk_size]
        print(f"  Processing chunk {i//chunk_size + 1}/{(len(core_contributors) + chunk_size - 1)//chunk_size}")

        try:
            chunk_results = _process_contributor_chunk(chunk, config, client, min_commits)

            # Merge results
            for repo_name, stats in chunk_results.items():
                if repo_name in all_repo_stats:
                    all_repo_stats[repo_name]['contributors'].update(stats['contributors'])
                    all_repo_stats[repo_name]['total_commits'] += stats['total_commits']
                else:
                    all_repo_stats[repo_name] = stats

        except Exception as e:
            print(f"    Chunk failed: {e}")
            continue

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

    # Simple query to get repo commits by contributor
    query = f"""
    SELECT
        CONCAT(p.artifact_namespace, '/', p.artifact_name) as repository_name,
        u.artifact_name as contributor,
        SUM(e.amount) as commits
    FROM int_events_daily__github AS e
    JOIN int_github_users AS u ON e.from_artifact_id = u.artifact_id
    JOIN artifacts_by_project_v1 AS p ON e.to_artifact_id = p.artifact_id
    WHERE u.artifact_name IN ('{contributors_str}')
      AND e.event_type = 'COMMIT_CODE'
      AND p.artifact_source = 'GITHUB'
      {bot_filter}
      {date_filter}
      {seed_exclusion}
    GROUP BY p.artifact_namespace, p.artifact_name, u.artifact_name
    HAVING SUM(e.amount) >= {min_commits}
    """

    df = client.to_pandas(query)
    if df.empty:
        return {}

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

    return repo_stats


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
        # Configure parameters from config.toml
        min_commits = config.get("analysis", {}).get("min_commits", 10)
        date_filter_days = config.get("general", {}).get("days_back", 0)

        # Use repo_to_contributors module
        extended_contributors_df = repo_to_contributors.get_contributors_by_repos(
            repo_identifiers=extended_repos,
            min_commits=min_commits,
            date_filter_days=date_filter_days,
            config=config
        )

        if extended_contributors_df.empty:
            print("No extended contributors found")
            return pd.DataFrame()

        # No limits - keep all contributors
        filtered_extended_contributors = extended_contributors_df

        # Save extended contributors data with custom filename
        output_dir = Path(config.get("general", {}).get("output_dir", "./raw"))
        extended_contrib_file = output_dir / "extended_contributors.csv"
        # Only save specified columns
        filtered_for_save = filtered_extended_contributors[['repository_name', 'contributor_handle']]

        # Save extended contributors data
        filtered_for_save.to_csv(extended_contrib_file, index=False)
        print(f"✓ Saved extended contributors: {extended_contrib_file}")

        return filtered_extended_contributors

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

        # Step 1: Load seed repositories from configuration
        seed_repos_file = output_dir / "seed_repos.csv"
        if seed_repos_file.exists():
            print("✓ Step 1: Seed repositories file already exists, loading from file...")
            seed_repos_df = pd.read_csv(seed_repos_file)
            seed_repos = seed_repos_df['repository_name'].tolist()
            print(f"✓ Loaded {len(seed_repos)} seed repositories from {seed_repos_file}")
        else:
            seed_repos = discover_seed_repositories(config)

        # Step 2: Find core contributors for seed repositories
        core_contrib_file = output_dir / "seed_contributors.csv"
        if core_contrib_file.exists():
            print("✓ Step 2: Core contributors file already exists, loading from file...")
            core_contrib_df = pd.read_csv(core_contrib_file)
            core_contributors = core_contrib_df['contributor_handle'].unique().tolist()
            print(f"✓ Loaded {len(core_contributors)} core contributors from {core_contrib_file}")
        else:
            core_contributors = find_core_contributors(seed_repos, config)

        # Check if extended analysis is enabled
        extended_analysis_enabled = config.get("analysis", {}).get("extended_analysis", True)

        if extended_analysis_enabled:
            print("📊 Extended analysis is enabled")

            # Step 3: Find extended repositories that core contributors contributed to
            extended_repos_file = output_dir / "extended_repos.csv"
            if extended_repos_file.exists():
                print("✓ Step 3: Extended repos file already exists, loading from file...")
                extended_repos_df = pd.read_csv(extended_repos_file)
                extended_repos = extended_repos_df['repository_name'].tolist()
                print(f"✓ Loaded {len(extended_repos)} extended repositories from {extended_repos_file}")
            else:
                print("🔍 Step 3: Finding extended repositories...")
                extended_repos = retry_extended_analysis(
                    find_extended_repos_by_stars,
                    core_contributors,
                    config
                )

            # Step 4: Find extended contributors from extended repositories
            extended_contrib_file = output_dir / "extended_contributors.csv"
            if extended_contrib_file.exists():
                print("✓ Step 4: Extended contributors file already exists, loading from file...")
                extended_contributors_df = pd.read_csv(extended_contrib_file)
                print(f"✓ Loaded {len(extended_contributors_df)} extended contributor records from {extended_contrib_file}")
            else:
                if extended_repos:
                    print("👥 Step 4: Finding extended contributors...")
                    extended_contributors_df = retry_extended_analysis(
                        find_extended_contributors,
                        extended_repos,
                        config
                    )
                else:
                    print("⚠️  No extended repositories found - skipping extended contributors")
                    extended_contributors_df = pd.DataFrame()
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
