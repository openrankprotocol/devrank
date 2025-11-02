#!/usr/bin/env python3
"""
GitHub Archive Processor

This script processes GitHub Archive files from archive/ directory,
extracts user-to-repo interactions, and saves them to cache/.

Usage:
    python process_archive.py [--archive-dir DIR]

    --archive-dir DIR: Directory containing archive files (default: archive)
"""

import os
import sys
import json
import gzip
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import calendar


def extract_repo_from_url(repo_url):
    """Extract owner/repo from GitHub URL"""
    if not repo_url:
        return None

    try:
        # Handle different URL formats
        if repo_url.startswith('https://github.com/'):
            path = repo_url.replace('https://github.com/', '')
        elif repo_url.startswith('https://api.github.com/repos/'):
            path = repo_url.replace('https://api.github.com/repos/', '')
        else:
            return None

        # Remove trailing slashes and extra path components
        parts = path.rstrip('/').split('/')
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    except:
        pass

    return None


def process_event(event, interactions):
    """Process a single GitHub event and extract user-repo interactions"""
    try:
        event_type = event.get('type', '')
        actor = event.get('actor', {})
        repo = event.get('repo', {})

        # Extract user info
        user_login = actor.get('login')
        if not user_login:
            return

        # Extract repo info
        repo_name = repo.get('name')
        if not repo_name:
            repo_url = repo.get('url')
            if repo_url:
                repo_name = extract_repo_from_url(repo_url)

        if not repo_name:
            return

        # Map GitHub event types to our interaction types
        event_mapping = {
            'PushEvent': 'COMMIT_CODE',
            'PullRequestEvent': 'PULL_REQUEST_OPENED',  # Will refine based on action
            'IssuesEvent': 'ISSUE_OPENED',  # Will refine based on action
            'WatchEvent': 'STARRED',
            'ForkEvent': 'FORKED',
            'CreateEvent': 'CREATED',
            'ReleaseEvent': 'RELEASE_PUBLISHED',
            'PullRequestReviewEvent': 'PULL_REQUEST_REVIEWED',
            'IssueCommentEvent': 'ISSUE_COMMENTED',
            'PullRequestReviewCommentEvent': 'PULL_REQUEST_COMMENTED'
        }

        mapped_event = event_mapping.get(event_type)

        # Handle special cases with payload inspection
        if event_type == 'PullRequestEvent':
            payload = event.get('payload', {})
            action = payload.get('action')
            if action == 'closed' and payload.get('pull_request', {}).get('merged'):
                mapped_event = 'PULL_REQUEST_MERGED'
            elif action == 'opened':
                mapped_event = 'PULL_REQUEST_OPENED'
            elif action == 'closed':
                mapped_event = 'PULL_REQUEST_CLOSED'
        elif event_type == 'IssuesEvent':
            payload = event.get('payload', {})
            action = payload.get('action')
            if action == 'opened':
                mapped_event = 'ISSUE_OPENED'
            elif action == 'closed':
                mapped_event = 'ISSUE_CLOSED'

        if not mapped_event:
            return

        # Calculate event count (for PushEvent, use commit count)
        event_count = 1
        if event_type == 'PushEvent':
            payload = event.get('payload', {})
            commits = payload.get('commits', [])
            event_count = len(commits) if commits else 1

        # Add to interactions
        interaction_key = (user_login, repo_name, mapped_event)
        interactions[interaction_key] += event_count

    except Exception as e:
        # Silently skip malformed events
        pass


def process_archive_file(file_path, interactions):
    """Process a single archive file"""
    events_processed = 0

    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    event = json.loads(line.strip())
                    process_event(event, interactions)
                    events_processed += 1
                except json.JSONDecodeError:
                    # Skip malformed JSON lines
                    continue
                except Exception:
                    # Skip other errors silently
                    continue
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return events_processed


def get_month_year_from_files(archive_files):
    """Extract month and year from archive files and verify they're consistent"""
    months_years = set()
    days_found = set()

    for file_path in archive_files:
        filename = file_path.name
        # Extract date from filename like "2015-01-01-0.json.gz"
        try:
            date_part = filename.split('-')
            if len(date_part) >= 4:  # year-month-day-hour.json.gz
                year = date_part[0]
                month = date_part[1]
                day = date_part[2]
                months_years.add((month, year))
                days_found.add(int(day))
        except:
            continue

    if len(months_years) == 0:
        raise ValueError("No valid date found in archive files")
    elif len(months_years) > 1:
        raise ValueError(f"Multiple months/years found in archive: {months_years}")

    month, year = months_years.pop()

    # Check if all days in the month are present
    year_int = int(year)
    month_int = int(month)
    days_in_month = calendar.monthrange(year_int, month_int)[1]
    expected_days = set(range(1, days_in_month + 1))

    missing_days = expected_days - days_found
    if missing_days:
        raise ValueError(f"Missing days in month {month}/{year}: {sorted(missing_days)}")

    return month, year


def main():
    parser = argparse.ArgumentParser(description='Process GitHub Archive files and extract interactions')
    parser.add_argument('--archive-dir', type=str, default='archive', help='Archive directory (default: archive)')

    args = parser.parse_args()

    print("GitHub Archive Processor")
    print("========================")
    print(f"Archive directory: {args.archive_dir}")

    # Check archive directory
    archive_dir = Path(args.archive_dir)
    if not archive_dir.exists():
        print(f"Error: Archive directory {archive_dir} does not exist")
        return 1

    # Get all .json.gz files
    archive_files = list(archive_dir.glob("*.json.gz"))
    if not archive_files:
        print("No .json.gz files found in archive directory")
        return 1

    print(f"Found {len(archive_files)} archive files")

    # Extract and verify month/year consistency
    try:
        month, year = get_month_year_from_files(archive_files)
        print(f"Processing files for month {month}/{year}")
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Process all files
    print("Processing archive files...")
    interactions = defaultdict(int)
    total_events = 0

    with tqdm(archive_files, desc="Processing files", unit="file") as pbar:
        for file_path in pbar:
            events_processed = process_archive_file(file_path, interactions)
            total_events += events_processed
            pbar.set_postfix({
                'events': f"{total_events:,}",
                'interactions': f"{len(interactions):,}"
            })

    print(f"✓ Processed {total_events:,} events")
    print(f"✓ Found {len(interactions):,} unique interactions")

    if not interactions:
        print("No interactions found to save")
        return 0

    # Convert interactions to DataFrame
    rows = []
    for (user, repo, event_type), event_count in interactions.items():
        rows.append({
            'user': user,
            'repo': repo,
            'event_type': event_type,
            'event_count': event_count
        })

    df = pd.DataFrame(rows)

    # Create cache directory
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    # Save to CSV
    output_file = cache_dir / f"interactions_{month}_{year}.csv"
    df.to_csv(output_file, index=False)

    print(f"✓ Saved {len(df):,} interactions to {output_file}")

    # Show top interactions
    print(f"\nTop 10 interactions:")
    print("=" * 80)
    top_interactions = df.nlargest(10, 'event_count')
    for _, row in top_interactions.iterrows():
        print(f"{row['user']:<20} {row['repo']:<30} {row['event_type']:<20} {row['event_count']:>6}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
