#!/usr/bin/env python3
"""
Extract Unique Repos and Users

This script extracts all unique repositories and users from CSV files in the raw/ directory
and saves them to filtered/repos.csv and filtered/users.csv respectively.
"""

import pandas as pd
from pathlib import Path
import os

def extract_unique_data():
    """Extract unique repositories and users from raw CSV files."""

    raw_dir = Path("raw")
    filtered_dir = Path("filtered")

    # Create filtered directory if it doesn't exist
    filtered_dir.mkdir(exist_ok=True)

    # Check if raw directory exists
    if not raw_dir.exists():
        print(f"ERROR: {raw_dir} directory not found!")
        return

    # Initialize sets to store unique values
    unique_repos = set()
    unique_users = set()

    # Process each CSV file in raw directory
    csv_files = list(raw_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {raw_dir}")
        return

    print(f"Processing {len(csv_files)} CSV files from {raw_dir}...")

    for csv_file in csv_files:
        print(f"  Processing {csv_file.name}...")

        try:
            df = pd.read_csv(csv_file)

            # Extract repository names
            if 'repository_name' in df.columns:
                repos = df['repository_name'].dropna().unique()
                unique_repos.update(repos)
                print(f"    Found {len(repos)} repositories")

            # Extract organization/repo_name combinations
            if 'organization' in df.columns and 'repo_name' in df.columns:
                org_repos = df.apply(lambda row: f"{row['organization']}/{row['repo_name']}"
                                   if pd.notna(row['organization']) and pd.notna(row['repo_name'])
                                   else None, axis=1).dropna().unique()
                unique_repos.update(org_repos)
                print(f"    Found {len(org_repos)} organization/repo combinations")

            # Extract contributor handles
            if 'contributor_handle' in df.columns:
                users = df['contributor_handle'].dropna().unique()
                unique_users.update(users)
                print(f"    Found {len(users)} contributors")

        except Exception as e:
            print(f"    ERROR processing {csv_file.name}: {e}")
            continue

    # Save unique repositories
    if unique_repos:
        repos_df = pd.DataFrame({'repository_name': sorted(unique_repos)})
        repos_file = filtered_dir / "repos.csv"
        repos_df.to_csv(repos_file, index=False)
        print(f"\n✓ Saved {len(unique_repos)} unique repositories to {repos_file}")
    else:
        print("\n✗ No repositories found")

    # Save unique users
    if unique_users:
        users_df = pd.DataFrame({'contributor_handle': sorted(unique_users)})
        users_file = filtered_dir / "users.csv"
        users_df.to_csv(users_file, index=False)
        print(f"✓ Saved {len(unique_users)} unique users to {users_file}")
    else:
        print("✗ No users found")

if __name__ == "__main__":
    extract_unique_data()
