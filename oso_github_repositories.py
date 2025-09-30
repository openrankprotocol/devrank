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

    if not config["general"]["seed_orgs"]:
        print("ERROR: No seed organizations specified in config.toml")
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

    seed_orgs = config["general"]["seed_orgs"]
    all_seed_repos = []

    print(f"Discovering repositories from {len(seed_orgs)} organizations...")

    for org in seed_orgs:
        print(f"  Finding repositories in {org} (with minimum {config.get('analysis', {}).get('min_core_contributors', 2)} core contributors)...")
        try:
            # Get filtering parameters
            min_core_contributors = config.get("analysis", {}).get("min_core_contributors", 2)
            min_commits = config.get("analysis", {}).get("min_commits", 10)
            days_back = config.get("general", {}).get("days_back", 0)

            # Build filter conditions
            filter_conditions = build_filter_conditions(config)
            bot_filter = f"AND {filter_conditions['bot_filter']}" if filter_conditions['bot_filter'] else ""

            # Add time filter if days_back is specified
            time_filter = ""
            if days_back > 0:
                from datetime import datetime, timedelta
                cutoff_date = datetime.now() - timedelta(days=days_back)
                time_filter = f"AND e.bucket_day >= DATE '{cutoff_date.strftime('%Y-%m-%d')}'"

            # Query to find repositories with minimum core contributors
            org_repos_query = f"""
            WITH contributor_commits AS (
                SELECT
                    p.artifact_id,
                    p.artifact_namespace,
                    p.artifact_name,
                    u.artifact_name as contributor,
                    SUM(e.amount) as total_commits
                FROM artifacts_by_project_v1 p
                JOIN int_events_daily__github e ON e.to_artifact_id = p.artifact_id
                JOIN int_github_users u ON e.from_artifact_id = u.artifact_id
                WHERE p.artifact_source = 'GITHUB'
                  AND p.artifact_namespace = '{org}'
                  AND e.event_type = 'COMMIT_CODE'
                  {bot_filter}
                  {time_filter}
                GROUP BY p.artifact_id, p.artifact_namespace, p.artifact_name, u.artifact_name
                HAVING SUM(e.amount) >= {min_commits}
            ),
            repo_contributor_counts AS (
                SELECT
                    artifact_namespace,
                    artifact_name,
                    CONCAT(artifact_namespace, '/', artifact_name) as repository_name,
                    COUNT(DISTINCT contributor) as core_contributor_count
                FROM contributor_commits
                GROUP BY artifact_namespace, artifact_name
                HAVING COUNT(DISTINCT contributor) >= {min_core_contributors}
            )
            SELECT
                artifact_namespace,
                artifact_name,
                repository_name
            FROM repo_contributor_counts
            ORDER BY artifact_name
            LIMIT {config.get('analysis', {}).get('max_repos_per_org', 200)}
            """

            org_repos_df = client.to_pandas(org_repos_query)

            if not org_repos_df.empty:
                org_repos = org_repos_df['repository_name'].tolist()
                all_seed_repos.extend(org_repos)
                print(f"    Found {len(org_repos)} repositories")
            else:
                print(f"    No repositories found for {org}")

        except Exception as e:
            # Handle pyoso exceptions that may have JSON parsing issues when converted to string
            try:
                error_msg = str(e)
            except Exception:
                error_msg = f"{type(e).__name__}: Unable to parse error message"
            print(f"    Error finding repositories for {org}: {error_msg}")
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
    Find extended repos by stars from orgs that contributors work with

    Given contributors:
    1. Find all repos they contributed to
    2. Find all unique orgs/users that own these repos
    3. Take top 10 repos from each org/user ranked by stars

    Args:
        core_contributors: List of contributor handles
        config: Configuration dict

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

    min_core_contributors = config.get("analysis", {}).get("min_core_contributors", 2)
    print(f"Finding repos contributed to by {len(core_contributors)} core contributors (with minimum {min_core_contributors} core contributors)...")

    # Build contributors filter
    contributors_str = "', '".join([c.replace("'", "''") for c in core_contributors])

    # Build date filter
    date_filter = ""
    date_filter_days = config.get("general", {}).get("days_back", 0)
    if date_filter_days > 0:
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=date_filter_days)
        date_filter = f"AND e.bucket_day >= DATE '{cutoff_date.strftime('%Y-%m-%d')}'"

    try:
        # Build filter conditions
        filter_conditions = build_filter_conditions(config)
        bot_filter = f"AND {filter_conditions['bot_filter']}" if filter_conditions['bot_filter'] else ""

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
          {bot_filter}
          {date_filter}
          AND p.artifact_source = 'GITHUB'
        """

        repos_df = client.to_pandas(repos_query)

        if repos_df.empty:
            print("✗ No repositories found for core contributors")
            return []

        print(f"✓ Found {len(repos_df)} repositories contributed to by core contributors")

        # Get unique organizations
        unique_orgs = repos_df['org_name'].unique()
        print(f"✓ Identified {len(unique_orgs)} unique organizations/users")

        # Process organizations in batches with parallel processing
        batch_size = 500
        batches = []

        # Create batches
        for batch_start in range(0, len(unique_orgs), batch_size):
            batch_orgs = unique_orgs[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            batches.append((batch_num, batch_orgs))

        print(f"Processing {len(batches)} organization batches sequentially...")

        from datetime import datetime
        start_time = datetime.now()

        # Process batches sequentially
        extended_repos = []

        for batch_num, batch_orgs in batches:
            import time
            time.sleep(0.5)  # Add delay to avoid rate limiting

            print(f"  Processing batch {batch_num}/{len(batches)} ({len(batch_orgs)} organizations)...")

            try:
                # Build filter conditions
                filter_conditions = build_filter_conditions(config)
                bot_filter = f"AND {filter_conditions['bot_filter']}" if filter_conditions['bot_filter'] else ""

                # Build org conditions for batch
                org_conditions = []
                for org in batch_orgs:
                    org_conditions.append(f"p.artifact_namespace = '{org.replace(chr(39), chr(39)+chr(39))}'")

                org_condition_str = " OR ".join(org_conditions)

                min_core_contributors = config.get("analysis", {}).get("min_core_contributors", 2)
                min_commits = config.get("analysis", {}).get("min_commits", 10)

                stars_query = f"""
                WITH contributor_commits AS (
                    SELECT
                        p.artifact_id,
                        p.artifact_namespace,
                        p.artifact_name,
                        u.artifact_name as contributor,
                        SUM(e.amount) as total_commits
                    FROM artifacts_by_project_v1 p
                    JOIN int_events_daily__github e ON e.to_artifact_id = p.artifact_id
                    JOIN int_github_users u ON e.from_artifact_id = u.artifact_id
                    WHERE ({org_condition_str})
                      AND e.event_type = 'COMMIT_CODE'
                      {bot_filter}
                      AND p.artifact_source = 'GITHUB'
                      {date_filter}
                    GROUP BY p.artifact_id, p.artifact_namespace, p.artifact_name, u.artifact_name
                    HAVING SUM(e.amount) >= {min_commits}
                ),
                repo_with_core_contributors AS (
                    SELECT
                        artifact_namespace,
                        artifact_name,
                        COUNT(DISTINCT contributor) as core_contributor_count
                    FROM contributor_commits
                    GROUP BY artifact_namespace, artifact_name
                    HAVING COUNT(DISTINCT contributor) >= {min_core_contributors}
                ),
                ranked_repos AS (
                    SELECT
                        p.artifact_namespace as org_name,
                        p.artifact_name as repo_name,
                        CONCAT(p.artifact_namespace, '/', p.artifact_name) as repository_name,
                        SUM(e.amount) as total_stars,
                        ROW_NUMBER() OVER (PARTITION BY p.artifact_namespace ORDER BY SUM(e.amount) DESC) as rn
                    FROM int_events_daily__github AS e
                    JOIN artifacts_by_project_v1 AS p ON e.to_artifact_id = p.artifact_id
                    JOIN repo_with_core_contributors rc ON p.artifact_namespace = rc.artifact_namespace AND p.artifact_name = rc.artifact_name
                    WHERE ({org_condition_str})
                      AND e.event_type = 'STARRED'
                      AND p.artifact_source = 'GITHUB'
                      {date_filter}
                    GROUP BY p.artifact_namespace, p.artifact_name
                )
                SELECT org_name, repo_name, repository_name, total_stars
                FROM ranked_repos
                WHERE rn <= {config.get('analysis', {}).get('max_repos_per_org', 200)}
                ORDER BY org_name, total_stars DESC
                """

                batch_stars_df = client.to_pandas(stars_query)

                if not batch_stars_df.empty:
                    batch_repos = batch_stars_df['repository_name'].tolist()
                    extended_repos.extend(batch_repos)

                    # Show summary by org
                    org_counts = batch_stars_df.groupby('org_name').size()
                    summary = []
                    for org, count in org_counts.items():
                        summary.append(f"{org}: {count}")
                    print(f"    Batch {batch_num}: Found repos for {', '.join(summary)}")
                else:
                    print(f"    Batch {batch_num}: No repositories found")

            except Exception as e:
                error_msg = str(e)
                if "Expecting value" in error_msg:
                    print(f"    ERROR: Batch {batch_num} received empty JSON response (likely rate limited)")
                else:
                    print(f"    ERROR: Batch {batch_num} query failed: {error_msg}")
                continue

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        extended_repos = list(set(extended_repos))  # Remove duplicates

        # No limits - find as many repos as possible
        output_dir = Path(config.get("general", {}).get("output_dir", "./raw"))

        print(f"✓ Found {len(extended_repos)} extended repositories in {duration:.1f}s")

        # Save extended repositories list
        output_dir.mkdir(parents=True, exist_ok=True)

        extended_repos_df = pd.DataFrame({
            'repository_name': extended_repos,
        })

        extended_file = output_dir / "crypto_extended_repos_by_stars.csv"
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
    extended_contrib_file = output_dir / "crypto_extended_contributors_by_stars.csv"
    # Only save specified columns
    filtered_for_save = filtered_extended_contributors[['repository_name', 'contributor_handle']]

    # Save extended contributors data
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
            extended_repos = find_extended_repos_by_stars(core_contributors, config)

        # Step 4: Find extended contributors using repo_to_contributors
        extended_contrib_file = output_dir / "crypto_extended_contributors_by_stars.csv"
        if extended_contrib_file.exists():
            print("✓ Step 4: Extended contributors file already exists, loading from file...")
            extended_contributors_df = pd.read_csv(extended_contrib_file)
            print(f"✓ Loaded {len(extended_contributors_df)} extended contributor records from {extended_contrib_file}")
        else:
            extended_contributors_df = find_extended_contributors(extended_repos, config)



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
