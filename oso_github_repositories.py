#!/usr/bin/env python3
"""
Crypto Ecosystem Contributor Network Analyzer - Configurable Version

This script analyzes the crypto development ecosystem by:
1. Starting with configurable crypto seed repositories
2. Finding core contributors to those repos
3. Mapping extended repositories they contribute to
4. Finding extended contributors in the broader network

Configuration is loaded from config.toml file.

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
from datetime import datetime, timedelta
from pathlib import Path

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
    """Validate configuration and provide defaults"""

    # Set defaults for missing sections
    defaults = {
        "general": {
            "output_dir": "./raw",
            "days_back": 0
        },
        "seed_repositories": {
            "repos": [
                "ethereum/go-ethereum",
                "bitcoin/bitcoin",
                "ethereum/solidity"
            ]
        },
        "contributors": {
            "min_commits": 5,
            "max_contributors_per_repo": 50,
            "max_core_contributors_for_extended_analysis": 20
        },
        "extended_repositories": {
            "min_core_contributors": 2,
            "max_extended_repos": 100,
            "max_extended_repos_for_contributors": 30
        },
        "extended_contributors": {
            "min_commits_extended": 3,
            "max_extended_contributors": 200
        },
        "query_limits": {
            "core_contributors_limit": 1000,
            "extended_repos_limit": 200,
            "extended_contributors_limit": 500
        },
        "filters": {
            "exclude_bots": False,
            "bot_keywords": ["bot", "dependabot", "mergify", "renovate", "github-actions"],
            "start_date": "",
            "end_date": ""
        },
        "output": {
            "timestamp_format": "%Y%m%d_%H%M%S",
            "file_prefix": "crypto",
            "include_headers": True,
            "include_timestamp_in_filename": True
        }
    }

    # Apply defaults for missing keys
    for section, section_defaults in defaults.items():
        if section not in config:
            config[section] = {}
        for key, default_value in section_defaults.items():
            if key not in config[section]:
                config[section][key] = default_value

    # Validate seed repositories
    if not config["seed_repositories"]["repos"]:
        print("ERROR: No seed repositories configured!")
        sys.exit(1)

    return config


def build_date_filter(config):
    """Build SQL date filter based on configuration"""
    date_filters = []

    # Days back filter
    if config["general"]["days_back"] > 0:
        cutoff_date = datetime.now() - timedelta(days=config["general"]["days_back"])
        date_filters.append(f"created_at >= DATE '{cutoff_date.strftime('%Y-%m-%d')}'")

    # Start date filter
    if config["filters"]["start_date"]:
        date_filters.append(f"created_at >= DATE '{config['filters']['start_date']}'")

    # End date filter
    if config["filters"]["end_date"]:
        date_filters.append(f"created_at <= DATE '{config['filters']['end_date']}'")

    return " AND " + " AND ".join(date_filters) if date_filters else ""


def build_bot_filter(config):
    """Build SQL filter to exclude bots if configured"""
    if not config["filters"]["exclude_bots"]:
        return ""

    bot_conditions = []
    for keyword in config["filters"]["bot_keywords"]:
        bot_conditions.append(f"LOWER(actor_login) NOT LIKE '%{keyword.lower()}%'")

    return " AND " + " AND ".join(bot_conditions) if bot_conditions else ""


def analyze_crypto_ecosystem(config_path="config.toml"):
    """
    Analyze crypto ecosystem by mapping contributor networks using TOML configuration.

    Args:
        config_path (str): Path to TOML configuration file

    Returns:
        dict: Dictionary containing all analysis results and CSV filenames
    """

    # Load and validate configuration
    config = load_config(config_path)
    config = validate_config(config)

    print(f"Starting CONFIGURABLE crypto ecosystem analysis...")
    print(f"Seed repositories: {len(config['seed_repositories']['repos'])}")
    print(f"Min commits for core contributors: {config['contributors']['min_commits']}")
    print(f"Date range: {config['general']['days_back']} days back" if config['general']['days_back'] > 0 else "Date range: All time")

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
        print("ERROR: Install pyoso with: pip install pyoso pandas python-dotenv tomli")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to initialize OSO client: {e}")
        sys.exit(1)

    timestamp = datetime.now().strftime(config["output"]["timestamp_format"])

    # Create output directory
    output_dir = Path(config["general"]["output_dir"])
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Build SQL filters
    date_filter = build_date_filter(config)
    bot_filter = build_bot_filter(config)

    # Step 1: Get seed repos and their top contributors
    print("\nStep 1: Finding seed repos and core contributors...")

    # Build repository filter
    seed_repos = config["seed_repositories"]["repos"]
    repo_filters = []
    for repo in seed_repos:
        if '/' in repo:
            repo_filters.append(f"repository_name = '{repo}'")
        else:
            print(f"WARNING: Invalid repo format '{repo}', should be 'org/repo'")

    if not repo_filters:
        print("ERROR: No valid seed repositories found!")
        return {}

    repo_filter_str = " OR ".join(repo_filters)

    try:
        # Single query to get top contributors across all seed repos
        core_contributors_query = f"""
        SELECT
            repository_name,
            actor_login as contributor_handle,
            actor_id,
            COUNT(*) as commit_count,
            COUNT(DISTINCT DATE(created_at)) as active_days,
            MIN(created_at) as first_commit,
            MAX(created_at) as last_commit
        FROM stg_github__commits
        WHERE ({repo_filter_str})
        AND actor_login IS NOT NULL
        AND actor_login != ''
        {date_filter}
        {bot_filter}
        GROUP BY repository_name, actor_login, actor_id
        HAVING COUNT(*) >= {config['contributors']['min_commits']}
        ORDER BY repository_name, COUNT(*) DESC
        LIMIT {config['query_limits']['core_contributors_limit']}
        """

        core_contributors_df = client.to_pandas(core_contributors_query)

        if not core_contributors_df.empty:
            print(f"✓ Found {len(core_contributors_df)} core contributor records")

            # Create seed repos summary
            seed_repos_summary = core_contributors_df.groupby('repository_name').agg({
                'contributor_handle': 'count',
                'commit_count': 'sum'
            }).reset_index()
            seed_repos_summary.columns = ['repository', 'contributor_count', 'total_commits']
            seed_repos_summary[['organization', 'repository_name']] = seed_repos_summary['repository'].str.split('/', expand=True)
            seed_repos_summary['status'] = 'found'

            # Get top contributors summary (aggregate across all seed repos)
            contributor_summary = core_contributors_df.groupby(['contributor_handle', 'actor_id']).agg({
                'commit_count': 'sum',
                'active_days': 'sum',
                'repository_name': lambda x: ', '.join(x.unique())
            }).reset_index()
            contributor_summary = contributor_summary.sort_values('commit_count', ascending=False)
            contributor_summary.columns = ['contributor_handle', 'actor_id', 'total_commits', 'total_active_days', 'seed_repositories']

            print(f"✓ Found {len(contributor_summary)} unique core contributors")

        else:
            print("✗ No core contributors found")
            seed_repos_summary = pd.DataFrame()
            contributor_summary = pd.DataFrame()

    except Exception as e:
        print(f"ERROR in Step 1: {e}")
        seed_repos_summary = pd.DataFrame()
        contributor_summary = pd.DataFrame()

    # Step 2: Find extended repositories
    print("\nStep 2: Finding extended repositories...")

    extended_repos_df = pd.DataFrame()

    if not contributor_summary.empty:
        try:
            # Take configured number of top contributors
            top_contributors = contributor_summary.head(config['contributors']['max_core_contributors_for_extended_analysis'])['contributor_handle'].tolist()
            contributors_str = "', '".join([c.replace("'", "''") for c in top_contributors])

            # Build exclusion filter for seed repos
            seed_repo_exclusions = "', '".join(seed_repos)

            extended_repos_query = f"""
            SELECT
                repository_name,
                COUNT(DISTINCT actor_login) as core_contributor_count,
                COUNT(*) as total_commits,
                MIN(created_at) as first_commit,
                MAX(created_at) as last_commit
            FROM stg_github__commits
            WHERE actor_login IN ('{contributors_str}')
            AND repository_name IS NOT NULL
            AND repository_name != ''
            AND repository_name LIKE '%/%'
            AND repository_name NOT IN ('{seed_repo_exclusions}')
            {date_filter}
            {bot_filter}
            GROUP BY repository_name
            HAVING COUNT(DISTINCT actor_login) >= {config['extended_repositories']['min_core_contributors']}
            ORDER BY COUNT(DISTINCT actor_login) DESC, COUNT(*) DESC
            LIMIT {config['extended_repositories']['max_extended_repos']}
            """

            extended_repos_df = client.to_pandas(extended_repos_query)

            if not extended_repos_df.empty:
                # Split repository name
                extended_repos_df[['organization', 'repository_name_clean']] = extended_repos_df['repository_name'].str.split('/', n=1, expand=True)

                # Apply configured limit
                extended_repos_df = extended_repos_df.head(config['extended_repositories']['max_extended_repos'])

                print(f"✓ Found {len(extended_repos_df)} extended repositories")
            else:
                print("✗ No extended repositories found")

        except Exception as e:
            print(f"ERROR in Step 2: {e}")
            extended_repos_df = pd.DataFrame()

    # Step 3: Find extended contributors
    print("\nStep 3: Finding extended contributors...")

    extended_contributors_df = pd.DataFrame()

    if not extended_repos_df.empty:
        try:
            # Take configured number of top extended repos
            top_extended_repos = extended_repos_df.head(config['extended_repositories']['max_extended_repos_for_contributors'])['repository_name'].tolist()
            repos_str = "', '".join([repo.replace("'", "''") for repo in top_extended_repos])

            extended_contributors_query = f"""
            SELECT
                actor_login as contributor_handle,
                actor_id,
                COUNT(DISTINCT repository_name) as repos_contributed,
                COUNT(*) as total_commits,
                MIN(created_at) as first_commit,
                MAX(created_at) as last_commit
            FROM stg_github__commits
            WHERE repository_name IN ('{repos_str}')
            AND actor_login IS NOT NULL
            AND actor_login != ''
            {date_filter}
            {bot_filter}
            GROUP BY actor_login, actor_id
            HAVING COUNT(*) >= {config['extended_contributors']['min_commits_extended']}
            ORDER BY COUNT(DISTINCT repository_name) DESC, COUNT(*) DESC
            LIMIT {config['extended_contributors']['max_extended_contributors']}
            """

            extended_contributors_df = client.to_pandas(extended_contributors_query)

            if not extended_contributors_df.empty:
                print(f"✓ Found {len(extended_contributors_df)} extended contributors")
            else:
                print("✗ No extended contributors found")

        except Exception as e:
            print(f"ERROR in Step 3: {e}")
            extended_contributors_df = pd.DataFrame()

    # Save results to CSV files
    print(f"\nSaving results to {output_dir}/...")
    saved_files = []

    file_prefix = config["output"]["file_prefix"]

    if config["output"]["include_timestamp_in_filename"]:
        timestamp_suffix = f"_{timestamp}"
    else:
        timestamp_suffix = ""

    # 1. Seed repositories
    if not seed_repos_summary.empty:
        seed_file = output_dir / f"{file_prefix}_seed_repos{timestamp_suffix}.csv"
        seed_output = seed_repos_summary[['organization', 'repository_name', 'contributor_count', 'total_commits', 'status']].copy()
        seed_output.to_csv(seed_file, index=False, header=config["output"]["include_headers"])
        saved_files.append(str(seed_file))
        print(f"✓ Saved seed repos: {seed_file}")

    # 2. Core contributors
    if not contributor_summary.empty:
        core_file = output_dir / f"{file_prefix}_core_contributors{timestamp_suffix}.csv"
        core_output = contributor_summary[['contributor_handle', 'total_commits', 'total_active_days', 'seed_repositories']].copy()
        core_output.to_csv(core_file, index=False, header=config["output"]["include_headers"])
        saved_files.append(str(core_file))
        print(f"✓ Saved core contributors: {core_file}")

    # 3. Extended repositories
    if not extended_repos_df.empty:
        extended_repos_file = output_dir / f"{file_prefix}_extended_repos{timestamp_suffix}.csv"
        extended_output = extended_repos_df[['organization', 'repository_name_clean', 'core_contributor_count', 'total_commits']].copy()
        extended_output.columns = ['organization', 'repository_name', 'core_contributor_count', 'total_commits']
        extended_output.to_csv(extended_repos_file, index=False, header=config["output"]["include_headers"])
        saved_files.append(str(extended_repos_file))
        print(f"✓ Saved extended repos: {extended_repos_file}")

    # 4. Extended contributors
    if not extended_contributors_df.empty:
        extended_contributors_file = output_dir / f"{file_prefix}_extended_contributors{timestamp_suffix}.csv"
        extended_contrib_output = extended_contributors_df[['contributor_handle', 'repos_contributed', 'total_commits']].copy()
        extended_contrib_output.to_csv(extended_contributors_file, index=False, header=config["output"]["include_headers"])
        saved_files.append(str(extended_contributors_file))
        print(f"✓ Saved extended contributors: {extended_contributors_file}")

    # Summary
    print(f"\n" + "="*60)
    print(f"CRYPTO ECOSYSTEM ANALYSIS COMPLETE")
    print(f"="*60)
    print(f"Configuration: {config_path}")
    print(f"Seed repositories: {len(seed_repos_summary)}")
    print(f"Core contributors: {len(contributor_summary)}")
    print(f"Extended repositories: {len(extended_repos_df)}")
    print(f"Extended contributors: {len(extended_contributors_df)}")
    print(f"\nFiles saved: {len(saved_files)}")
    for file in saved_files:
        print(f"  - {file}")

    # Show sample data
    if not contributor_summary.empty:
        print(f"\nTop 5 Core Contributors:")
        for i, (_, contrib) in enumerate(contributor_summary.head(5).iterrows(), 1):
            repos = contrib['seed_repositories'][:50] + "..." if len(contrib['seed_repositories']) > 50 else contrib['seed_repositories']
            print(f"  {i}. {contrib['contributor_handle']} - {contrib['total_commits']} commits")

    if not extended_repos_df.empty:
        print(f"\nTop 5 Extended Repositories:")
        for i, (_, repo) in enumerate(extended_repos_df.head(5).iterrows(), 1):
            print(f"  {i}. {repo['organization']}/{repo['repository_name_clean']} - {repo['core_contributor_count']} core contributors")

    return {
        'config': config,
        'seed_repos': seed_repos_summary,
        'core_contributors': contributor_summary,
        'extended_repos': extended_repos_df,
        'extended_contributors': extended_contributors_df,
        'files': saved_files,
        'timestamp': timestamp
    }


if __name__ == "__main__":
    try:
        # Check for custom config file argument
        config_file = sys.argv[1] if len(sys.argv) > 1 else "config.toml"
        results = analyze_crypto_ecosystem(config_file)
    except KeyboardInterrupt:
        print("\nCancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Failed: {e}")
        sys.exit(1)
