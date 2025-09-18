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
    print("\n" + "="*60)
    print("STEP 1: DISCOVERING SEED REPOSITORIES")
    print("="*60)

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
        print("ERROR: Install pyoso with: pip install pyoso pandas python-dotenv tomli")
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
                # Apply max repos per org limit
                max_repos = config.get("seed_repositories", {}).get("max_repos_per_org", 0)
                if max_repos > 0:
                    org_repos_df = org_repos_df.head(max_repos)

                org_repos = org_repos_df['repository_name'].tolist()
                all_seed_repos.extend(org_repos)
                print(f"    Found {len(org_repos)} repositories")
            else:
                print(f"    No repositories found for {org}")

        except Exception as e:
            print(f"    Error finding repositories for {org}: {e}")
            continue

    print(f"\n✓ Total seed repositories discovered: {len(all_seed_repos)}")

    if not all_seed_repos:
        print("ERROR: No seed repositories found!")
        sys.exit(1)

    # Save seed repositories list
    output_dir = Path(config.get("general", {}).get("output_dir", "./raw"))
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_repos_df = pd.DataFrame({
        'repository_name': all_seed_repos,
        'step': 'seed_discovery',
        'discovery_time': datetime.now().isoformat()
    })
    seed_repos_df[['organization', 'repo_name']] = seed_repos_df['repository_name'].str.split('/', expand=True)

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
    print("\n" + "="*60)
    print("STEP 2: FINDING CORE CONTRIBUTORS")
    print("="*60)

    print(f"Finding contributors for {len(seed_repos)} seed repositories...")

    # Configure parameters from config.toml
    min_commits = config.get("contributors", {}).get("min_commits", 10)
    date_filter_days = config.get("general", {}).get("days_back", 0)

    print(f"Configuration:")
    print(f"  Minimum commits: {min_commits}")
    print(f"  Date filter: {'All time' if date_filter_days == 0 else f'Last {date_filter_days} days'}")

    # Use repo_to_contributors module
    contributors_df = repo_to_contributors.get_contributors_by_repos(
        repo_identifiers=seed_repos,
        min_commits=min_commits,
        date_filter_days=date_filter_days
    )

    if contributors_df.empty:
        print("ERROR: No contributors found for seed repositories")
        return []

    # Analyze and save contributors data
    repo_to_contributors.analyze_contributors_data(contributors_df)
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

    print(f"\n✓ Selected {len(core_contributor_handles)} core contributors for extended analysis")
    print("Top 5 core contributors:")
    for i, (contributor, data) in enumerate(top_contributors.head(5).iterrows(), 1):
        print(f"  {i}. {contributor} - {data['total_commits']:,} commits across {data['repository_name']} repos")

    return core_contributor_handles


def find_extended_ecosystem(core_contributors, config):
    """
    Step 3: Find extended organizations and repositories using contributors_to_repos.py

    Returns:
        tuple: (extended_repos_list, extended_organizations_list)
    """
    print("\n" + "="*60)
    print("STEP 3: FINDING EXTENDED ECOSYSTEM")
    print("="*60)

    print(f"Finding repositories for {len(core_contributors)} core contributors...")

    # Configure parameters from config.toml
    min_commits = config.get("extended_repositories", {}).get("min_core_contributors", 2)
    include_org_repos = True  # Always include organization repos
    date_filter_days = config.get("general", {}).get("days_back", 0)

    print(f"Configuration:")
    print(f"  Minimum commits: {min_commits}")
    print(f"  Include organization repos: {include_org_repos}")
    print(f"  Date filter: {'All time' if date_filter_days == 0 else f'Last {date_filter_days} days'}")

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

    # Analyze and save repos data
    contributors_to_repos.analyze_repos_data(repos_data)
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

    print(f"\n✓ Extended ecosystem discovered:")
    print(f"  Extended repositories: {len(extended_repos)}")
    print(f"  Extended organizations: {len(extended_organizations)}")

    return extended_repos, extended_organizations


def find_extended_contributors(extended_repos, config):
    """
    Step 4: Find extended contributors from extended repositories using repo_to_contributors.py

    Returns:
        pd.DataFrame: Extended contributors data
    """
    print("\n" + "="*60)
    print("STEP 4: FINDING EXTENDED CONTRIBUTORS")
    print("="*60)

    if not extended_repos:
        print("No extended repositories to analyze")
        return pd.DataFrame()

    print(f"Finding contributors for {len(extended_repos)} extended repositories...")

    # Limit repositories to avoid overwhelming the system
    max_repos_for_contributors = config.get("extended_repositories", {}).get("max_extended_repos_for_contributors", 100)
    if len(extended_repos) > max_repos_for_contributors:
        print(f"Limiting to {max_repos_for_contributors} repositories for contributor analysis")
        extended_repos = extended_repos[:max_repos_for_contributors]

    # Configure parameters from config.toml
    min_commits = config.get("extended_contributors", {}).get("min_commits_extended", 3)
    date_filter_days = config.get("general", {}).get("days_back", 0)

    print(f"Configuration:")
    print(f"  Repositories to analyze: {len(extended_repos)}")
    print(f"  Minimum commits: {min_commits}")
    print(f"  Date filter: {'All time' if date_filter_days == 0 else f'Last {date_filter_days} days'}")

    # Use repo_to_contributors module
    extended_contributors_df = repo_to_contributors.get_contributors_by_repos(
        repo_identifiers=extended_repos,
        min_commits=min_commits,
        date_filter_days=date_filter_days
    )

    if extended_contributors_df.empty:
        print("No extended contributors found")
        return pd.DataFrame()

    # Apply configured limit for extended contributors
    max_extended_contributors = config.get("extended_contributors", {}).get("max_extended_contributors", 500)

    # Prioritize contributors by total commits across all repos
    contributor_summary = extended_contributors_df.groupby('contributor_handle').agg({
        'total_commits': 'sum',
        'repository_name': 'nunique',
        'active_days': 'sum'
    }).sort_values('total_commits', ascending=False).head(max_extended_contributors)

    # Filter original data to include only top contributors
    top_contributor_handles = contributor_summary.index.tolist()
    filtered_extended_contributors = extended_contributors_df[
        extended_contributors_df['contributor_handle'].isin(top_contributor_handles)
    ]

    # Save extended contributors data with custom filename
    output_dir = Path(config.get("general", {}).get("output_dir", "./raw"))
    extended_contrib_file = output_dir / "crypto_extended_contributors.csv"
    filtered_extended_contributors.to_csv(extended_contrib_file, index=False)

    print(f"\n✓ Extended contributors analysis complete:")
    print(f"  Total contributor records: {len(filtered_extended_contributors):,}")
    print(f"  Unique contributors: {len(top_contributor_handles):,}")
    print(f"  Unique repositories: {filtered_extended_contributors['repository_name'].nunique():,}")
    print(f"✓ Saved: {extended_contrib_file}")

    # Show top extended contributors
    print("\nTop 5 extended contributors:")
    for i, (contributor, data) in enumerate(contributor_summary.head(5).iterrows(), 1):
        print(f"  {i}. {contributor} - {data['total_commits']:,} commits across {data['repository_name']} repos")

    return filtered_extended_contributors


def generate_final_summary(config, seed_repos, core_contributors, extended_repos, extended_contributors_df):
    """Generate final summary and save consolidated results."""

    print("\n" + "="*80)
    print("CRYPTO ECOSYSTEM ANALYSIS COMPLETE")
    print("="*80)

    # Summary statistics
    print(f"FINAL SUMMARY:")
    print(f"  Seed organizations: {len(config['seed_organizations']['orgs'])}")
    print(f"  Seed repositories: {len(seed_repos)}")
    print(f"  Core contributors: {len(core_contributors)}")
    print(f"  Extended repositories: {len(extended_repos)}")
    print(f"  Extended contributors: {extended_contributors_df['contributor_handle'].nunique() if not extended_contributors_df.empty else 0}")

    # List all generated files
    output_dir = Path(config.get("general", {}).get("output_dir", "./raw"))
    generated_files = []

    for file_pattern in ["crypto_seed_repos.csv", "repo_contributors.csv", "contributor_repos.csv",
                        "organization_repos.csv", "crypto_extended_contributors.csv"]:
        file_path = output_dir / file_pattern
        if file_path.exists():
            generated_files.append(str(file_path))

    print(f"\nGenerated files ({len(generated_files)}):")
    for file_path in generated_files:
        try:
            file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
            print(f"  📄 {file_path} ({file_size:.1f} MB)")
        except:
            print(f"  📄 {file_path}")

    # Create a final summary file
    summary_data = {
        'analysis_timestamp': datetime.now().isoformat(),
        'config_file': 'config.toml',
        'seed_organizations_count': len(config['seed_organizations']['orgs']),
        'seed_repositories_count': len(seed_repos),
        'core_contributors_count': len(core_contributors),
        'extended_repositories_count': len(extended_repos),
        'extended_contributors_count': extended_contributors_df['contributor_handle'].nunique() if not extended_contributors_df.empty else 0,
        'total_files_generated': len(generated_files)
    }

    summary_df = pd.DataFrame([summary_data])
    summary_file = output_dir / "crypto_analysis_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✓ Analysis summary saved: {summary_file}")

    return generated_files


def main(config_path="config.toml"):
    """Main orchestration function."""

    print("Crypto Ecosystem Contributor Network Analyzer - Pipeline Version")
    print("="*80)
    print("Multi-step pipeline using specialized analysis modules")
    print(f"Configuration: {config_path}")

    # Load and validate configuration
    config = load_config(config_path)
    config = validate_config(config)

    print(f"\nPipeline Configuration:")
    print(f"  Seed organizations: {len(config['seed_organizations']['orgs'])}")
    print(f"  Output directory: {config.get('general', {}).get('output_dir', './raw')}")
    date_filter_days = config.get('general', {}).get('days_back', 0)
    date_filter_text = 'All time' if date_filter_days == 0 else f'{date_filter_days} days back'
    print(f"  Date filter: {date_filter_text}")

    try:
        # Step 1: Discover seed repositories from organizations
        seed_repos = discover_seed_repositories(config)

        # Step 2: Find core contributors using repo_to_contributors
        core_contributors = find_core_contributors(seed_repos, config)

        # Step 3: Find extended ecosystem using contributors_to_repos
        extended_repos, extended_organizations = find_extended_ecosystem(core_contributors, config)

        # Step 4: Find extended contributors using repo_to_contributors
        extended_contributors_df = find_extended_contributors(extended_repos, config)

        # Generate final summary
        generated_files = generate_final_summary(
            config, seed_repos, core_contributors, extended_repos, extended_contributors_df
        )

        print(f"\n🎉 SUCCESS! Crypto ecosystem analysis pipeline completed successfully!")
        print(f"📊 Generated {len(generated_files)} data files ready for analysis")

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
