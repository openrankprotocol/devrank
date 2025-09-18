#!/usr/bin/env python3
"""
Contributors to Repositories Mapper

This script takes contributor IDs and finds:
1. All repositories they have contributed to
2. All other repositories within the same organizations (if any)

Uses the Open Source Observer (OSO) database.

Prerequisites:
1. Create account at www.opensource.observer
2. Generate API key in Account Settings > API Keys
3. Set OSO_API_KEY environment variable or in .env file
4. Install dependencies: pip install pyoso pandas python-dotenv

Usage:
    python contributors_to_repos.py
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


def get_repos_by_contributors(contributor_identifiers, min_commits=1, include_org_repos=True, date_filter_days=0):
    """
    Find all repositories for given contributors, plus organization repositories.

    Args:
        contributor_identifiers (list): List of contributor handles (GitHub usernames)
        min_commits (int): Minimum number of commits to be considered a contribution
        include_org_repos (bool): Whether to include all repos from same organizations
        date_filter_days (int): Number of days back to consider (0 = all time)

    Returns:
        dict: Contains two DataFrames:
            - 'contributed_repos': Repositories the contributors directly worked on
            - 'organization_repos': All repositories from the same organizations
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

    print(f"Finding repositories for {len(contributor_identifiers)} contributors...")

    # Build date filter
    date_filter = ""
    if date_filter_days > 0:
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=date_filter_days)
        date_filter = f"AND e.bucket_day >= DATE '{cutoff_date.strftime('%Y-%m-%d')}'"

    # Build contributor conditions
    contributors_str = "', '".join([c.replace("'", "''") for c in contributor_identifiers])

    try:
        # Step 1: Find repositories that contributors have directly worked on
        print("Step 1: Finding repositories contributors have worked on...")

        contributed_repos_query = f"""
        SELECT
            CONCAT(p.artifact_namespace, '/', p.artifact_name) as repository_name,
            p.artifact_namespace as organization,
            p.artifact_name as repo_name,
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
          AND u.artifact_name IN ('{contributors_str}')
          AND u.artifact_name IS NOT NULL
          AND u.artifact_name != ''
          AND p.artifact_namespace IS NOT NULL
          AND p.artifact_name IS NOT NULL
          {date_filter}
        GROUP BY p.artifact_namespace, p.artifact_name, u.artifact_name, u.artifact_id
        HAVING SUM(e.amount) >= {min_commits}
        ORDER BY p.artifact_namespace, p.artifact_name, SUM(e.amount) DESC
        """

        contributed_repos_df = client.to_pandas(contributed_repos_query)

        if contributed_repos_df.empty:
            print("✗ No repositories found for the specified contributors")
            return {'contributed_repos': pd.DataFrame(), 'organization_repos': pd.DataFrame()}

        print(f"✓ Found {len(contributed_repos_df)} direct contribution records")

        # Step 2: Find all organizations these contributors work with
        organizations = contributed_repos_df['organization'].unique().tolist()
        print(f"✓ Identified {len(organizations)} organizations: {', '.join(organizations[:5])}{'...' if len(organizations) > 5 else ''}")

        organization_repos_df = pd.DataFrame()

        if include_org_repos and organizations:
            print("Step 2: Finding all repositories in these organizations...")

            # Build organization conditions
            org_conditions = []
            for org in organizations:
                org_conditions.append(f"artifact_namespace = '{org}'")

            org_condition_str = " OR ".join(org_conditions)

            org_repos_query = f"""
            SELECT DISTINCT
                artifact_namespace as organization,
                artifact_name as repo_name,
                CONCAT(artifact_namespace, '/', artifact_name) as repository_name,
                artifact_id,
                artifact_source_id
            FROM artifacts_by_project_v1
            WHERE
              artifact_source = 'GITHUB'
              AND ({org_condition_str})
              AND artifact_namespace IS NOT NULL
              AND artifact_name IS NOT NULL
            ORDER BY artifact_namespace, artifact_name
            """

            organization_repos_df = client.to_pandas(org_repos_query)

            if not organization_repos_df.empty:
                print(f"✓ Found {len(organization_repos_df)} total repositories across all organizations")
            else:
                print("✗ No organization repositories found")

        return {
            'contributed_repos': contributed_repos_df,
            'organization_repos': organization_repos_df
        }

    except Exception as e:
        print(f"ERROR: Query failed: {e}")
        return {'contributed_repos': pd.DataFrame(), 'organization_repos': pd.DataFrame()}


def save_repos_data(repos_data, output_dir="./raw"):
    """Save repositories data to CSV files."""

    contributed_repos_df = repos_data['contributed_repos']
    organization_repos_df = repos_data['organization_repos']

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # Save contributed repositories
    if not contributed_repos_df.empty:
        contrib_file = output_path / "contributor_repos.csv"
        contributed_repos_df.to_csv(contrib_file, index=False)
        saved_files.append(str(contrib_file))
        print(f"✓ Saved contributed repos: {contrib_file}")
        print(f"  Records: {len(contributed_repos_df)}")

    # Save organization repositories
    if not organization_repos_df.empty:
        org_file = output_path / "organization_repos.csv"
        organization_repos_df.to_csv(org_file, index=False)
        saved_files.append(str(org_file))
        print(f"✓ Saved organization repos: {org_file}")
        print(f"  Records: {len(organization_repos_df)}")

    return saved_files


def analyze_repos_data(repos_data):
    """Analyze and display repositories data."""

    contributed_repos_df = repos_data['contributed_repos']
    organization_repos_df = repos_data['organization_repos']

    print(f"\n" + "="*60)
    print(f"REPOSITORIES ANALYSIS")
    print(f"="*60)

    if not contributed_repos_df.empty:
        # Contributed repositories stats
        total_contrib_records = len(contributed_repos_df)
        unique_contrib_repos = contributed_repos_df['repository_name'].nunique()
        unique_contributors = contributed_repos_df['contributor_handle'].nunique()
        total_commits = contributed_repos_df['total_commits'].sum()

        print(f"DIRECTLY CONTRIBUTED REPOSITORIES:")
        print(f"  Total contribution records: {total_contrib_records:,}")
        print(f"  Unique repositories: {unique_contrib_repos:,}")
        print(f"  Unique contributors: {unique_contributors:,}")
        print(f"  Total commits: {total_commits:,}")

        # Top repositories by commits
        top_repos = contributed_repos_df.groupby('repository_name').agg({
            'total_commits': 'sum',
            'contributor_handle': 'nunique'
        }).sort_values('total_commits', ascending=False).head(10)

        print(f"\n  Top 10 Repositories (by commits):")
        for i, (repo, data) in enumerate(top_repos.iterrows(), 1):
            print(f"    {i:2d}. {repo} - {data['total_commits']:,} commits, {data['contributor_handle']} contributors")

        # Top contributors
        top_contributors = contributed_repos_df.groupby('contributor_handle').agg({
            'total_commits': 'sum',
            'repository_name': 'nunique'
        }).sort_values('total_commits', ascending=False).head(10)

        print(f"\n  Top 10 Contributors (by commits):")
        for i, (contributor, data) in enumerate(top_contributors.iterrows(), 1):
            print(f"    {i:2d}. {contributor} - {data['total_commits']:,} commits across {data['repository_name']} repos")

    if not organization_repos_df.empty:
        # Organization repositories stats
        total_org_repos = len(organization_repos_df)
        unique_orgs = organization_repos_df['organization'].nunique()

        print(f"\nORGANIZATION REPOSITORIES:")
        print(f"  Total repositories: {total_org_repos:,}")
        print(f"  Unique organizations: {unique_orgs:,}")

        # Repositories by organization
        org_counts = organization_repos_df['organization'].value_counts().head(10)

        print(f"\n  Top 10 Organizations (by repository count):")
        for i, (org, count) in enumerate(org_counts.items(), 1):
            print(f"    {i:2d}. {org} - {count} repositories")


def main():
    """Main function with example usage."""

    print("Contributors to Repositories Mapper")
    print("="*50)

    # Example contributor identifiers - modify these as needed
    example_contributors = [
        "vitalik",
        "gakonst",
        "mattsse",
        "chriseth",
        "fjl"
    ]

    print(f"Example: Finding repositories for {len(example_contributors)} contributors")
    print("Contributors:")
    for contributor in example_contributors:
        print(f"  - {contributor}")

    # Configuration
    min_commits = 3  # Minimum commits to be considered a contribution
    include_org_repos = True  # Include all repos from same organizations
    date_filter_days = 0  # 0 = all time, 365 = last year, etc.

    print(f"\nConfiguration:")
    print(f"  Minimum commits: {min_commits}")
    print(f"  Include organization repos: {include_org_repos}")
    print(f"  Date filter: {'All time' if date_filter_days == 0 else f'Last {date_filter_days} days'}")

    # Find repositories
    repos_data = get_repos_by_contributors(
        contributor_identifiers=example_contributors,
        min_commits=min_commits,
        include_org_repos=include_org_repos,
        date_filter_days=date_filter_days
    )

    contributed_repos = repos_data['contributed_repos']
    organization_repos = repos_data['organization_repos']

    if not contributed_repos.empty or not organization_repos.empty:
        # Analyze data
        analyze_repos_data(repos_data)

        # Save data
        saved_files = save_repos_data(repos_data)

        print(f"\n✅ Process completed successfully!")
        print(f"Files saved:")
        for file_path in saved_files:
            print(f"  - {file_path}")
    else:
        print("❌ No repositories data found")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Failed: {e}")
        sys.exit(1)
