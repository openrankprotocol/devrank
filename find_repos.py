#!/usr/bin/env python3
"""
GitHub Stars Ranking for Solana Repositories using OSO

This script loads repositories from raw/Solana-export.csv, fetches their GitHub stars
using the Open Source Observer (OSO) database, and displays the top 100 repositories
sorted by star count.

Prerequisites:
1. Create account at www.opensource.observer
2. Generate API key in Account Settings > API Keys
3. Set OSO_API_KEY environment variable or in .env file
4. Install dependencies: pip install pyoso pandas python-dotenv

Usage:
    python github_stars_ranking.py [--test] [--top N] [--limit N]

    --test: Run filtering tests only
    --top N: Show top N repositories (default: 100)
    --limit N: Limit number of repositories to process (for testing)
"""

import os
import sys
import pandas as pd
import time
import argparse
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from collections import deque

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def extract_repo_info(github_url):
    """Extract owner and repo name from GitHub URL"""
    try:
        parsed = urlparse(github_url.strip())
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) >= 2 and parsed.netloc == 'github.com':
            return path_parts[0], path_parts[1]
    except:
        pass
    return None, None

def should_filter_repo(repo_name):
    """Check if repo should be filtered out based on keywords"""
    filter_keywords = ['docs', 'example', 'examples', 'frontend', 'ui', 'test', 'tutorial', 'dapp']
    repo_lower = repo_name.lower()

    # Split repo name by common separators to check for whole words
    words = re.split(r'[-_.]|(?<=[a-z])(?=[A-Z])', repo_name)
    words = [w.lower() for w in words if w]  # Convert to lowercase and remove empty strings

    for keyword in filter_keywords:
        # Check if keyword appears as a whole word
        if keyword in words:
            return True
        # Also check for keywords at the start or end of the repo name
        if repo_lower.startswith(keyword + '-') or repo_lower.startswith(keyword + '_'):
            return True
        if repo_lower.endswith('-' + keyword) or repo_lower.endswith('_' + keyword):
            return True
        # Check for exact matches in the full name (case insensitive)
        if keyword == repo_lower:
            return True
        # Check for plural forms and common variations
        if keyword == 'example' and 'examples' in words:
            return True
        if keyword == 'test' and any(w.startswith('test') for w in words if len(w) > 4):
            return True
    return False

def test_filtering():
    """Test the filtering logic with sample repository names"""
    test_cases = [
        # Should be filtered
        ("solana-docs", True),
        ("example-program", True),
        ("solana-examples", True),
        ("my-frontend", True),
        ("wallet-ui", True),
        ("test-suite", True),
        ("solana-tutorial", True),
        ("my-dapp", True),
        ("ExampleRepo", True),
        ("testingTools", True),
        # Should NOT be filtered
        ("solana-core", False),
        ("jupiter-swap", False),
        ("wallet-adapter", False),
        ("serum-dex", False),
        ("anchor-lang", False),
        ("solana-program-library", False),
        ("metaplex", False),
        ("contest-winner", False),  # contains 'test' but not as separate word
        ("manifest-trading", False),  # contains 'test' substring but not as word
    ]

    print("Testing filtering logic:")
    all_passed = True
    for repo_name, expected in test_cases:
        result = should_filter_repo(repo_name)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {repo_name}: {result} (expected {expected})")
        if result != expected:
            all_passed = False

    if all_passed:
        print("All filtering tests passed!")
    else:
        print("Some filtering tests failed!")

    return all_passed

def load_solana_repos(limit=None):
    """Load repository URLs from Solana-export.csv"""
    csv_path = Path("raw/Solana-export.csv")

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
        if 'url' not in df.columns:
            print("ERROR: 'url' column not found in CSV")
            sys.exit(1)

        # Limit rows if specified
        if limit:
            df = df.head(limit)
            print(f"Processing first {limit} entries for testing")

        repos = []
        filtered_repos = []
        filtered_count = 0
        invalid_count = 0

        for idx, row in df.iterrows():
            url = row['url']
            owner, repo = extract_repo_info(url)
            if owner and repo:
                if should_filter_repo(repo):
                    filtered_count += 1
                    filtered_repos.append(f"{owner}/{repo}")
                    continue
                repos.append({
                    'url': url,
                    'owner': owner,
                    'repo': repo,
                    'full_name': f"{owner}/{repo}"
                })
            else:
                invalid_count += 1

        print(f"Loaded {len(repos)} valid GitHub repositories from {len(df)} total entries")
        print(f"Filtered out {filtered_count} repositories containing docs/examples/frontend/ui/test/tutorial/dapp keywords")
        print(f"Skipped {invalid_count} invalid URLs")

        # Show some examples of filtered repositories
        if filtered_repos:
            print(f"Examples of filtered repositories:")
            for example in filtered_repos[:10]:  # Show first 10 examples
                print(f"  - {example}")
            if len(filtered_repos) > 10:
                print(f"  ... and {len(filtered_repos) - 10} more")

        return repos

    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        sys.exit(1)

def chunk_list(lst, chunk_size):
    """Split list into chunks of specified size"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def load_cache_stars_and_commits():
    """Load cached STARRED and COMMIT_CODE events from all cache files"""
    cache_dir = Path("cache")

    repo_stars = {}
    repo_commits = {}
    total_interactions = 0
    files_processed = 0

    # Find all cache files
    cache_files = list(cache_dir.glob("interactions_*.csv"))

    if not cache_files:
        print(f"Warning: No cache files found in {cache_dir}. Run generate_trust.py first to create cache.")
        return repo_stars, repo_commits

    print(f"Found {len(cache_files)} cache files to process")

    for cache_file in cache_files:
        try:
            df = pd.read_csv(cache_file)
            total_interactions += len(df)
            files_processed += 1
            print(f"✓ Processing {cache_file.name} with {len(df)} interactions")

            # Filter for STARRED and COMMIT_CODE events
            starred_df = df[df['event_type'] == 'STARRED']
            commits_df = df[df['event_type'] == 'COMMIT_CODE']

            # Aggregate stars by repository
            if not starred_df.empty:
                repo_star_counts = starred_df.groupby('repo')['event_count'].sum()
                for repo, count in repo_star_counts.items():
                    repo_stars[repo] = repo_stars.get(repo, 0) + count

            # Aggregate commits by repository
            if not commits_df.empty:
                repo_commit_counts = commits_df.groupby('repo')['event_count'].sum()
                for repo, count in repo_commit_counts.items():
                    repo_commits[repo] = repo_commits.get(repo, 0) + count

        except Exception as e:
            print(f"Warning: Error loading {cache_file}: {e}")

    print(f"✓ Processed {files_processed} cache files with {total_interactions} total interactions")
    print(f"✓ Loaded {len(repo_stars)} repositories with star data from cache")
    print(f"✓ Loaded {len(repo_commits)} repositories with commit data from cache")



    return repo_stars, repo_commits

def ensure_commit_events_loaded(cached_commits):
    """Ensure commit events are loaded from cache, load if not already done"""
    if cached_commits:
        print("✓ Commit events already loaded from cache")
        return cached_commits

    print("→ Loading commit events from all cache files...")
    cache_dir = Path("cache")
    cache_files = list(cache_dir.glob("interactions_*.csv"))

    repo_commits = {}
    for cache_file in cache_files:
        try:
            df = pd.read_csv(cache_file)
            commits_df = df[df['event_type'] == 'COMMIT_CODE']

            if not commits_df.empty:
                repo_commit_counts = commits_df.groupby('repo')['event_count'].sum()
                for repo, count in repo_commit_counts.items():
                    repo_commits[repo] = repo_commits.get(repo, 0) + count
        except Exception as e:
            print(f"Warning: Error loading commit events from {cache_file}: {e}")

    print(f"✓ Loaded {len(repo_commits)} repositories with commit data from all cache files")
    return repo_commits

def get_repo_stars_batch(repo_list, client, cached_stars, cached_commits):
    """Get star counts for a batch of repositories using OSO (repo_list contains only uncached repos)"""
    if not repo_list:
        return []

    # Query OSO for all repositories in the batch
    repo_conditions = []
    for repo_info in repo_list:
        owner = repo_info['owner']
        repo = repo_info['repo']
        repo_conditions.append(f"(artifact_namespace = '{owner}' AND artifact_name = '{repo}')")

    condition_str = ' OR '.join(repo_conditions)

    # Query for commit counts only (stars come from cache) - limit to last 365 days
    query = f"""
    SELECT
        p.artifact_namespace,
        p.artifact_name,
        CONCAT(p.artifact_namespace, '/', p.artifact_name) as full_name,
        COALESCE(SUM(e.amount), 0) as total_commits
    FROM artifacts_v1 p
    LEFT JOIN int_events_daily__github e ON p.artifact_id = e.to_artifact_id
    WHERE p.artifact_source = 'GITHUB'
      AND ({condition_str})
      AND (e.bucket_day IS NULL OR e.bucket_day >= CURRENT_DATE - INTERVAL '365' DAY)
    GROUP BY p.artifact_namespace, p.artifact_name
    """

    try:
        df = client.to_pandas(query)
        print(f"    OSO query returned {len(df)} rows for {len(repo_list)} repositories")

        # Convert results to dictionary for easy lookup
        repo_data = {}
        for _, row in df.iterrows():
            full_name = row['full_name']
            repo_data[full_name] = {
                'commits': int(row['total_commits']) if pd.notna(row['total_commits']) else 0,
                'artifact_namespace': row['artifact_namespace'],
                'artifact_name': row['artifact_name']
            }

        # Process all repos - add cached stars if available
        results = []
        for repo_info in repo_list:
            full_name = repo_info['full_name']
            if full_name in repo_data:
                data = repo_data[full_name]
                # Get stars from cache if available
                cached_star_count = cached_stars.get(full_name, 0)
                results.append({
                    'repo_org': data['artifact_namespace'],
                    'repo_name': data['artifact_name'],
                    'stars': cached_star_count,
                    'commits': data['commits'],
                    'status': 'found'
                })
            else:
                # Check if we have stars in cache even if no commits
                cached_star_count = cached_stars.get(full_name, 0)
                results.append({
                    'repo_org': repo_info['owner'],
                    'repo_name': repo_info['repo'],
                    'stars': cached_star_count,
                    'commits': 0,
                    'status': 'found' if cached_star_count > 0 else 'not_found'
                })

        return results

    except Exception as e:
        print(f"Error querying OSO for batch: {e}")
        # Return error results for all repos in batch
        return [{
            'repo_org': repo_info['owner'],
            'repo_name': repo_info['repo'],
            'stars': 0,
            'commits': 0,
            'status': 'error'
        } for repo_info in repo_list]

def fetch_all_stars_oso(repos):
    """Fetch star counts for all repositories using cache first, then OSO for missing data"""
    # Load cache first
    print("Loading cache for STARRED and COMMIT_CODE events from all cache files...")
    cached_stars, cached_commits = load_cache_stars_and_commits()

    # Ensure commit events are loaded
    cached_commits = ensure_commit_events_loaded(cached_commits)

    # Filter out repos already fully cached
    uncached_repos = []
    cached_results = []

    print(f"Checking {len(repos)} repositories against cache...")

    for repo in repos:
        full_name = repo['full_name']
        if full_name in cached_stars or full_name in cached_commits:
            # Found in cache
            stars = cached_stars.get(full_name, 0)
            commits = cached_commits.get(full_name, 0)

            org_name = full_name.split('/')
            if len(org_name) == 2:
                cached_results.append({
                    'repo_org': org_name[0],
                    'repo_name': org_name[1],
                    'stars': stars,
                    'commits': commits,
                    'status': 'found'
                })
                print(f"  ✓ Found in cache: {full_name} ({stars} stars, {commits} commits)")
            else:
                uncached_repos.append(repo)
        else:
            uncached_repos.append(repo)

    print(f"✓ Found {len(cached_results)} repositories in cache")
    print(f"→ {len(uncached_repos)} repositories need OSO queries")

    # If all repos are cached, return cache results
    if not uncached_repos:
        return cached_results

    # Get OSO API key for uncached repos
    api_key = os.getenv('OSO_API_KEY')
    if not api_key:
        print("ERROR: OSO API key required. Set OSO_API_KEY environment variable.")
        print("Get key at: https://www.opensource.observer")
        sys.exit(1)

    # Initialize OSO client
    try:
        import pyoso
        client = pyoso.Client(api_key=api_key)
        print("✓ OSO client initialized")
    except ImportError:
        print("ERROR: Install pyoso with: pip install pyoso pandas python-dotenv")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to initialize OSO client: {e}")
        sys.exit(1)

    results = cached_results.copy()  # Start with cached results
    total_repos = len(uncached_repos)
    batch_size = 200

    print(f"Fetching data for {total_repos} uncached repositories using OSO (batch size: {batch_size})...")

    # Create job queue for uncached repos only
    job_queue = deque()
    for i, repo_batch in enumerate(chunk_list(uncached_repos, batch_size)):
        batch_num = i + 1
        total_batches = (total_repos + batch_size - 1) // batch_size
        job_queue.append((batch_num, repo_batch, total_batches))

    completed_batches = 0

    # Process jobs in queue - move to next batch if one fails
    while job_queue:
        time.sleep(0.5)  # Add delay to avoid rate limiting

        # Take one job from queue
        current_job = job_queue.popleft()
        batch_num, repo_batch, total_batches = current_job

        # Progress indicator
        progress_pct = (completed_batches / total_batches) * 100 if total_batches > 0 else 100
        queue_size = len(job_queue) + 1  # +1 for current job

        print(f"Processing batch {batch_num}/{total_batches} ({len(repo_batch)} repos) - {progress_pct:.1f}% complete, {queue_size} in queue")

        try:
            batch_results = get_repo_stars_batch(repo_batch, client, cached_stars, cached_commits)

            # Success statistics
            found_count = sum(1 for r in batch_results if r['status'] == 'found')
            not_found_count = sum(1 for r in batch_results if r['status'] == 'not_found')
            error_count = sum(1 for r in batch_results if r['status'] == 'error')

            results.extend(batch_results)
            completed_batches += 1

            print(f"  ✓ Batch {batch_num} completed: {found_count} found, {not_found_count} not found, {error_count} errors")

        except Exception as e:
            try:
                error_msg = str(e)
            except Exception:
                error_msg = f"{type(e).__name__}: Unable to parse error message"

            print(f"  ✗ Batch {batch_num} failed: {error_msg}")
            print(f"  → Adding batch {batch_num} back to queue for retry")

            # Add failed batch back to the end of the queue
            job_queue.append(current_job)
            continue

    print(f"✓ Completed processing {completed_batches}/{total_batches} batches")

    # Final statistics
    total_found = sum(1 for r in results if r['status'] == 'found')
    total_not_found = sum(1 for r in results if r['status'] == 'not_found')
    total_errors = sum(1 for r in results if r['status'] == 'error')
    cached_count = len(cached_results)
    print(f"Final results: {total_found} found ({cached_count} from cache), {total_not_found} not found, {total_errors} errors")
    return results

def display_top_repos(results, top_n=100):
    """Display top N repositories by star count"""
    # Filter out errors and not found repos, then sort by stars
    valid_results = [r for r in results if r['status'] == 'found' and r['stars'] >= 0]
    sorted_results = sorted(valid_results, key=lambda x: x['stars'], reverse=True)

    print(f"\n{'='*100}")
    print(f"TOP {min(top_n, len(sorted_results))} SOLANA REPOSITORIES BY GITHUB STARS")
    print(f"{'='*100}")
    print(f"{'Rank':<6} {'Stars':<8} {'Commits':<10} {'Repository':<35}")
    print(f"{'-'*100}")

    for i, repo in enumerate(sorted_results[:top_n], 1):
        stars_str = f"{repo['stars']:,}"
        commits_str = f"{repo['commits']:,}"
        repo_name = f"{repo['repo_org']}/{repo['repo_name']}"
        print(f"{i:<6} {stars_str:<8} {commits_str:<10} {repo_name:<35}")

    # Summary statistics
    print(f"\n{'='*100}")
    print("SUMMARY STATISTICS")
    print(f"{'='*100}")

    total_repos = len(results)
    found_repos = len([r for r in results if r['status'] == 'found'])
    not_found = len([r for r in results if r['status'] == 'not_found'])
    errors = len([r for r in results if r['status'] == 'error'])

    print(f"Total repositories processed: {total_repos:,}")
    print(f"Successfully found: {found_repos:,} ({found_repos/total_repos*100:.1f}%)")
    print(f"Not found in OSO: {not_found:,} ({not_found/total_repos*100:.1f}%)")
    print(f"Errors: {errors:,} ({errors/total_repos*100:.1f}%)")

    if valid_results:
        total_stars = sum(r['stars'] for r in valid_results)
        total_commits = sum(r['commits'] for r in valid_results)
        avg_stars = total_stars / len(valid_results)
        avg_commits = total_commits / len(valid_results)
        max_stars = max(r['stars'] for r in valid_results)
        median_stars = sorted([r['stars'] for r in valid_results])[len(valid_results)//2]

        print(f"\nRepository statistics (found repositories only):")
        print(f"Total stars: {total_stars:,}")
        print(f"Total commits: {total_commits:,}")
        print(f"Average stars: {avg_stars:.1f}")
        print(f"Average commits: {avg_commits:.1f}")
        print(f"Median stars: {median_stars:,}")
        print(f"Maximum stars: {max_stars:,}")

def save_results(results):
    """Save results to CSV file in raw/ directory"""
    raw_dir = Path("raw")
    raw_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = raw_dir / f"solana_github_stars_oso_{timestamp}.csv"

    # Filter to only successful results and select only required columns
    successful_results = [r for r in results if r['status'] == 'found']
    simplified_results = [{
        'repo_org': r['repo_org'],
        'repo_name': r['repo_name'],
        'stars': r['stars'],
        'commits': r['commits']
    } for r in successful_results]

    if simplified_results:
        df = pd.DataFrame(simplified_results)
        df = df.sort_values('stars', ascending=False)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    else:
        # Create empty CSV with headers
        df = pd.DataFrame(columns=['repo_org', 'repo_name', 'stars', 'commits'])
        df.to_csv(output_file, index=False)
        print(f"\nEmpty results saved to: {output_file}")



def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='GitHub Stars Ranking for Solana Repositories using OSO')
    parser.add_argument('--test', action='store_true', help='Run filtering tests only')
    parser.add_argument('--top', type=int, default=100, help='Number of top repositories to display (default: 100)')
    parser.add_argument('--limit', type=int, help='Limit number of repositories to process (for testing)')

    args = parser.parse_args()

    if args.test:
        test_filtering()
        return

    print("GitHub Stars Ranking for Solana Repositories using OSO")
    print("=" * 60)

    # Load repositories
    repos = load_solana_repos(limit=args.limit)

    if not repos:
        print("No repositories found after filtering. Exiting.")
        return

    # Fetch star counts using OSO
    results = fetch_all_stars_oso(repos)

    # Display top repositories
    display_top_repos(results, top_n=args.top)

    # Save results
    save_results(results)

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
