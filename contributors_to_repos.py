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


def get_repos_by_contributors(contributor_identifiers, min_commits=1, include_org_repos=True, date_filter_days=0, config=None):
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
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=date_filter_days)
        date_filter = f"AND e.bucket_day >= DATE '{cutoff_date.strftime('%Y-%m-%d')}'"

    # Build filter conditions from config
    filter_conditions = build_filter_conditions(config or {})
    bot_filter = f"AND {filter_conditions['bot_filter']}" if filter_conditions['bot_filter'] else ""

    # Build contributor conditions
    contributors_str = "', '".join([c.replace("'", "''") for c in contributor_identifiers])

    try:
        # Step 1: Find repositories that contributors have directly worked on

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
          {bot_filter}
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
        # Only save specified columns
        filtered_df = contributed_repos_df[['organization', 'repo_name']]
        filtered_df.to_csv(contrib_file, index=False)
        saved_files.append(str(contrib_file))

    # Save organization repositories
    if not organization_repos_df.empty:
        org_file = output_path / "organization_repos.csv"
        # Only save specified columns
        filtered_org_df = organization_repos_df[['organization', 'repo_name']]
        filtered_org_df.to_csv(org_file, index=False)
        saved_files.append(str(org_file))

    return saved_files
