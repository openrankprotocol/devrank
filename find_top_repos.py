#!/usr/bin/env python3
"""
Script to find top repositories from ecosystems based on stars and commits.

Usage:
    python find_repos.py path/to/ecosystems/file1.csv [file2.csv ...]
    python find_repos.py ecosystems/*.csv

This script:
1. Loads repositories from the specified ecosystems file(s)
2. Loads all interaction data from cache/ directory
3. Applies exponential time decay to scores based on data age
4. Normalizes stars, commits, and unique contributors (0-1 scale)
5. Calculates weighted score: 0.4*stars + 0.4*contributors + 0.2*commits
6. Saves top 100 repos per ecosystem to raw/[ecosystem_name]_top_100.csv
7. Merges all results into raw/merged_top_repos.csv with unique repos
"""

import csv
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import toml


def parse_github_url(url):
    """
    Extract owner/repo from GitHub URL.

    Args:
        url: GitHub URL like "https://github.com/owner/repo"

    Returns:
        String in format "owner/repo" or None if invalid
    """
    try:
        # Remove quotes if present
        url = url.strip('"').strip("'")

        # Parse URL
        parsed = urlparse(url)
        if parsed.netloc not in ["github.com", "www.github.com"]:
            return None

        # Get path parts
        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) >= 2:
            return f"{path_parts[0]}/{path_parts[1]}"

        return None
    except Exception:
        return None


def load_config(config_path="config.toml"):
    """
    Load configuration from TOML file.

    Args:
        config_path: Path to config file

    Returns:
        Dictionary with config data
    """
    try:
        with open(config_path, "r") as f:
            config = toml.load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: {config_path} not found, no filtering will be applied")
        return {}


def is_bot(username, bot_keywords):
    """
    Check if username matches bot patterns.

    Args:
        username: Username to check
        bot_keywords: List of bot keywords

    Returns:
        True if username is a bot, False otherwise
    """
    if not username:
        return False

    username_lower = username.lower()

    # Check for exact [bot] suffix
    if username_lower.endswith("[bot]") or username_lower.endswith("-bot"):
        return True

    # Check for bot keywords
    for keyword in bot_keywords:
        if keyword.lower() in username_lower:
            return True

    return False


def is_test_repo(repo_name, repo_exclude_keywords):
    """
    Check if repository matches test/docs/example patterns.

    Args:
        repo_name: Repository name to check
        repo_exclude_keywords: List of keywords to exclude

    Returns:
        True if repo should be excluded, False otherwise
    """
    if not repo_name:
        return False

    repo_lower = repo_name.lower()

    # Check for exclude keywords in repo name
    for keyword in repo_exclude_keywords:
        if keyword.lower() in repo_lower:
            return True

    return False


def load_ecosystems_file(ecosystems_file):
    """
    Load repositories from ecosystems CSV file.

    Args:
        ecosystems_file: Path to ecosystems CSV file

    Returns:
        Dictionary mapping repo names to sub_ecosystem, and ecosystem name
    """
    repos = {}
    ecosystem_name = Path(ecosystems_file).stem

    with open(ecosystems_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("url", "")
            sub_ecosystem = row.get("sub_ecosystems", "").strip('"').strip("'")

            repo_name = parse_github_url(url)
            if repo_name:
                repos[repo_name] = sub_ecosystem

    return repos, ecosystem_name


def calculate_time_decay(month, year, decay_rate=0.1):
    """
    Calculate exponential time decay factor based on data age.

    Args:
        month: Month of the data (1-12)
        year: Year of the data
        decay_rate: Decay rate per month (default 0.1 = 10% per month)

    Returns:
        Decay factor between 0 and 1
    """
    # Get current date
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month

    # Calculate months difference
    months_ago = (current_year - year) * 12 + (current_month - month)

    # Apply exponential decay: decay_factor = e^(-decay_rate * months_ago)
    decay_factor = math.exp(-decay_rate * months_ago)

    return decay_factor


def load_cache_data(cache_dir, target_repos, config, verbose=True):
    """
    Load interaction data from all cache files for target repositories.
    Applies exponential time decay based on data age.

    Args:
        cache_dir: Path to cache directory
        target_repos: Set of repository names to track
        config: Configuration dictionary with filters
        verbose: Whether to print progress messages

    Returns:
        Dictionary mapping repo names to {'stars': count, 'commits': count, 'contributors': dict}
    """
    repo_stats = defaultdict(lambda: {"stars": 0, "commits": 0, "contributors": {}})
    cache_path = Path(cache_dir)

    # Get filtering config
    filters = config.get("filters", {})
    exclude_bots = filters.get("exclude_bots", True)
    bot_keywords = filters.get("bot_keywords", [])
    exclude_test_repos = filters.get("exclude_test_repos", True)
    repo_exclude_keywords = filters.get("repo_exclude_keywords", [])

    # Find all interaction CSV files
    cache_files = list(cache_path.glob("interactions_*.csv"))

    if verbose:
        print(f"Found {len(cache_files)} cache files to process...")

    for i, cache_file in enumerate(cache_files, 1):
        # Extract month and year from filename: interactions_MM_YYYY.csv
        filename = cache_file.stem  # Gets filename without extension
        parts = filename.split("_")

        try:
            month = int(parts[1])
            year = int(parts[2])
            decay_factor = calculate_time_decay(month, year)
        except (IndexError, ValueError):
            # If parsing fails, use no decay
            decay_factor = 1.0
            print(
                f"Warning: Could not parse date from {cache_file.name}, using no decay"
            )

        print(
            f"Processing {cache_file.name} ({i}/{len(cache_files)}) - decay factor: {decay_factor:.4f}..."
        )

        with open(cache_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                repo = row.get("repo", "")
                user = row.get("user", "")
                event_type = row.get("event_type", "")
                event_count = int(row.get("event_count", 0))

                # Only process repos in our target ecosystem
                if repo not in target_repos:
                    continue

                # Filter out bots if enabled
                if exclude_bots and bot_keywords and is_bot(user, bot_keywords):
                    continue

                # Filter out test/docs/example repos if enabled
                if (
                    exclude_test_repos
                    and repo_exclude_keywords
                    and is_test_repo(repo, repo_exclude_keywords)
                ):
                    continue

                # Count stars, commits, and contributors with time decay applied
                if event_type == "STARRED":
                    repo_stats[repo]["stars"] += event_count * decay_factor
                elif event_type == "COMMIT_CODE":
                    repo_stats[repo]["commits"] += event_count * decay_factor
                    # Track commit count per user (for contributor threshold)
                    if user not in repo_stats[repo]["contributors"]:
                        repo_stats[repo]["contributors"][user] = 0
                    repo_stats[repo]["contributors"][user] += event_count * decay_factor

    return repo_stats


def normalize_values(values):
    """
    Normalize values to 0-1 range.

    Args:
        values: List of numeric values

    Returns:
        List of normalized values (0-1 range)
    """
    if not values:
        return []

    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        # All values are the same
        return [1.0] * len(values)

    return [(v - min_val) / (max_val - min_val) for v in values]


def calculate_scores(
    repo_stats, star_weight=0.4, contributor_weight=0.4, commit_weight=0.2
):
    """
    Calculate weighted scores for repositories with normalized metrics.

    Args:
        repo_stats: Dictionary of repo stats
        star_weight: Weight for stars (default 0.4)
        contributor_weight: Weight for unique contributors (default 0.4)
        commit_weight: Weight for commits (default 0.2)

    Returns:
        List of tuples (repo_name, score, stars, commits, contributors) sorted by score descending
    """
    if not repo_stats:
        return []

    # Extract raw values
    repos = list(repo_stats.keys())
    stars_list = [repo_stats[repo]["stars"] for repo in repos]
    commits_list = [repo_stats[repo]["commits"] for repo in repos]
    # Count contributors with >3 commits
    contributors_list = [
        sum(
            1
            for commit_count in repo_stats[repo]["contributors"].values()
            if commit_count > 3
        )
        for repo in repos
    ]

    # Normalize all metrics to 0-1 range
    normalized_stars = normalize_values(stars_list)
    normalized_commits = normalize_values(commits_list)
    normalized_contributors = normalize_values(contributors_list)

    # Calculate weighted scores
    scored_repos = []
    for i, repo in enumerate(repos):
        score = (
            (star_weight * normalized_stars[i])
            + (contributor_weight * normalized_contributors[i])
            + (commit_weight * normalized_commits[i])
        )
        scored_repos.append(
            (
                repo,
                score,
                int(stars_list[i]),
                int(commits_list[i]),
                contributors_list[i],
            )
        )

    # Sort by score descending
    scored_repos.sort(key=lambda x: x[1], reverse=True)

    return scored_repos


def save_top_repos(
    scored_repos, ecosystem_repos, ecosystem_name, output_dir, top_n=100
):
    """
    Save top N repositories to CSV file.

    Args:
        scored_repos: List of scored repositories
        ecosystem_repos: Dictionary mapping repo names to sub_ecosystems
        ecosystem_name: Name of the ecosystem
        output_dir: Directory to save output file
        top_n: Number of top repos to save (default 100)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"{ecosystem_name}_top_100.csv"

    # Take top N repos
    top_repos = scored_repos[:top_n]

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "repo",
                "score",
                "stars",
                "commits",
                "contributors",
                "sub_ecosystem",
            ]
        )

        for rank, (repo, score, stars, commits, contributors) in enumerate(
            top_repos, 1
        ):
            sub_ecosystem = ecosystem_repos.get(repo, "")
            writer.writerow(
                [
                    rank,
                    repo,
                    f"{score:.4f}",
                    stars,
                    commits,
                    contributors,
                    sub_ecosystem,
                ]
            )

    print(f"\nSaved top {len(top_repos)} repos to {output_file}")

    # Print summary
    if top_repos:
        print(f"\nTop 10 repositories:")
        print(
            f"{'Rank':<6} {'Repository':<50} {'Score':<12} {'Stars':<10} {'Commits':<10} {'Contributors':<15}"
        )
        print("-" * 113)
        for rank, (repo, score, stars, commits, contributors) in enumerate(
            top_repos[:10], 1
        ):
            print(
                f"{rank:<6} {repo:<50} {score:<12.4f} {stars:<10} {commits:<10} {contributors:<15}"
            )


def process_ecosystem_file(ecosystems_file, cache_dir, raw_dir, config, verbose=True):
    """
    Process a single ecosystem file.

    Args:
        ecosystems_file: Path to ecosystem file
        cache_dir: Path to cache directory
        raw_dir: Path to raw output directory
        config: Configuration dictionary with filters
        verbose: Whether to print detailed progress

    Returns:
        True if successful, False otherwise
    """
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Processing: {ecosystems_file}")
        print(f"{'=' * 80}")

    # Verify file exists
    if not Path(ecosystems_file).exists():
        print(f"Error: File not found: {ecosystems_file}")
        return False

    if verbose:
        print(f"Loading ecosystems file: {ecosystems_file}")

    ecosystem_repos, ecosystem_name = load_ecosystems_file(ecosystems_file)

    if verbose:
        print(
            f"Loaded {len(ecosystem_repos)} repositories from {ecosystem_name} ecosystem"
        )

    if not ecosystem_repos:
        print(f"No valid repositories found in {ecosystems_file}")
        return False

    if verbose:
        print(f"\nLoading interaction data from cache directory: {cache_dir}")

    repo_stats = load_cache_data(
        cache_dir, set(ecosystem_repos.keys()), config, verbose=verbose
    )

    # Filter repos that have at least some activity
    active_repos = {
        repo: stats
        for repo, stats in repo_stats.items()
        if stats["stars"] > 0 or stats["commits"] > 0
    }

    if verbose:
        print(f"\nFound {len(active_repos)} repositories with activity")

    if not active_repos:
        print(f"No repositories with activity found for {ecosystem_name}")
        return False

    if verbose:
        print(
            "\nCalculating weighted scores (0.4*stars + 0.4*contributors + 0.2*commits)..."
        )

    scored_repos = calculate_scores(active_repos)

    if verbose:
        print(f"Saving top 100 repos to {raw_dir}/...")

    save_top_repos(scored_repos, ecosystem_repos, ecosystem_name, raw_dir)

    return True


def merge_all_results(raw_dir, output_filename="merged_top_repos.csv"):
    """
    Merge all ecosystem top 100 files into one file with unique repos.

    Args:
        raw_dir: Directory containing the ecosystem top 100 files
        output_filename: Name of the merged output file
    """
    print(f"\n{'=' * 80}")
    print(f"Merging all results into {output_filename}...")
    print(f"{'=' * 80}")

    raw_path = Path(raw_dir)
    all_repos = {}

    # Find all top_100 CSV files
    top_100_files = list(raw_path.glob("*_top_100.csv"))

    if not top_100_files:
        print("No top_100.csv files found to merge")
        return

    print(f"Found {len(top_100_files)} files to merge")

    for csv_file in top_100_files:
        ecosystem_name = csv_file.stem.replace("_top_100", "")
        print(f"  Loading {csv_file.name}...")

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                repo = row["repo"]
                score = float(row["score"])
                stars = int(row["stars"])
                commits = int(row["commits"])
                contributors = int(row.get("contributors", 0))
                sub_ecosystem = row.get("sub_ecosystem", "")

                # If repo exists, keep the one with higher score
                if repo in all_repos:
                    if score > all_repos[repo]["score"]:
                        # Update with higher score, but append ecosystem
                        ecosystems = all_repos[repo]["ecosystems"]
                        if ecosystem_name not in ecosystems:
                            ecosystems.append(ecosystem_name)
                        all_repos[repo] = {
                            "score": score,
                            "stars": stars,
                            "commits": commits,
                            "contributors": contributors,
                            "sub_ecosystem": sub_ecosystem,
                            "ecosystems": ecosystems,
                        }
                    else:
                        # Keep existing entry but add ecosystem
                        if ecosystem_name not in all_repos[repo]["ecosystems"]:
                            all_repos[repo]["ecosystems"].append(ecosystem_name)
                else:
                    # New repo
                    all_repos[repo] = {
                        "score": score,
                        "stars": stars,
                        "commits": commits,
                        "contributors": contributors,
                        "sub_ecosystem": sub_ecosystem,
                        "ecosystems": [ecosystem_name],
                    }

    # Sort by score descending
    sorted_repos = sorted(all_repos.items(), key=lambda x: x[1]["score"], reverse=True)

    # Write merged file
    output_file = raw_path / output_filename
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "repo",
                "score",
                "stars",
                "commits",
                "contributors",
                "sub_ecosystem",
                "ecosystems",
            ]
        )

        for rank, (repo, data) in enumerate(sorted_repos, 1):
            ecosystems_str = ";".join(data["ecosystems"])
            writer.writerow(
                [
                    rank,
                    repo,
                    f"{data['score']:.4f}",
                    data["stars"],
                    data["commits"],
                    data["contributors"],
                    data["sub_ecosystem"],
                    ecosystems_str,
                ]
            )

    print(f"\nMerged {len(all_repos)} unique repos into {output_file}")
    print(f"\nTop 10 repos from merged results:")
    print(
        f"{'Rank':<6} {'Repository':<50} {'Score':<12} {'Stars':<10} {'Commits':<10} {'Contributors':<15}"
    )
    print("-" * 113)
    for rank, (repo, data) in enumerate(sorted_repos[:10], 1):
        print(
            f"{rank:<6} {repo:<50} {data['score']:<12.4f} {int(data['stars']):<10} {int(data['commits']):<10} {int(data['contributors']):<15}"
        )


def main():
    """Main function to process ecosystem file(s) and find top repos."""
    if len(sys.argv) < 2:
        print("Usage: python find_repos.py path/to/ecosystems/file.csv [file2.csv ...]")
        print("   or: python find_repos.py ecosystems/*.csv")
        sys.exit(1)

    ecosystems_files = sys.argv[1:]

    # Get script directory to find cache and raw directories
    script_dir = Path(__file__).parent
    cache_dir = script_dir / "cache"
    raw_dir = script_dir / "raw"

    if not cache_dir.exists():
        print(f"Error: Cache directory not found: {cache_dir}")
        sys.exit(1)

    # Load configuration
    config_path = script_dir / "config.toml"
    config = load_config(config_path)

    # Print filter configuration
    filters = config.get("filters", {})
    if filters:
        print(f"\nFiltering configuration:")
        print(f"  Exclude bots: {filters.get('exclude_bots', False)}")
        print(
            f"  Bot keywords: {len(filters.get('bot_keywords', []))} keywords configured"
        )
        print(f"  Exclude test/docs repos: {filters.get('exclude_test_repos', False)}")
        print(
            f"  Repo exclude keywords: {len(filters.get('repo_exclude_keywords', []))} keywords configured"
        )
        print()

    print(f"Processing {len(ecosystems_files)} ecosystem file(s)...")

    successful = 0
    failed = 0

    for ecosystems_file in ecosystems_files:
        try:
            if process_ecosystem_file(
                ecosystems_file,
                cache_dir,
                raw_dir,
                config,
                verbose=True,
            ):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {ecosystems_file}: {e}")
            failed += 1

    print(f"\n{'=' * 80}")
    print(f"Processing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'=' * 80}")

    # Merge all results into one file
    if successful > 0:
        merge_all_results(raw_dir)


if __name__ == "__main__":
    main()
