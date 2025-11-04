#!/usr/bin/env python3
"""
Generate trust relationships from cached GitHub interactions.

This script:
1. Reads all interaction files from cache/ directory one at a time
2. Applies weights from config.toml to event counts
3. Generates both user-to-repo and repo-to-user trust relationships
4. Appends results directly to trust/github.csv without aggregation
5. Processes files incrementally to minimize memory usage

Memory-efficient approach: processes one file at a time and appends to output.
"""

import pandas as pd
from pathlib import Path
import toml
from typing import Dict, Set
import os


def load_target_repos(ecosystems_dir: Path = Path("ecosystems")) -> Set[str]:
    """
    Load all target repositories from ecosystem CSV files.

    Extracts repo names (owner/repo) from GitHub URLs in all CSV files
    in the ecosystems/ directory and returns them as a set.

    Args:
        ecosystems_dir: Path to the ecosystems directory

    Returns:
        Set of repository names in format "owner/repo"
    """
    target_repos = set()

    if not ecosystems_dir.exists():
        print(
            f"Warning: {ecosystems_dir} not found, no ecosystem filtering will be applied"
        )
        return target_repos

    csv_files = list(ecosystems_dir.glob("*.csv"))

    if not csv_files:
        print(f"Warning: No CSV files found in {ecosystems_dir}")
        return target_repos

    print(f"Loading target repos from {len(csv_files)} ecosystem file(s)...")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            if "url" not in df.columns:
                print(f"Warning: 'url' column not found in {csv_file.name}, skipping")
                continue

            # Extract repo names from GitHub URLs
            for url in df["url"].dropna():
                url = str(url).strip()
                # Handle GitHub URLs: https://github.com/owner/repo
                if "github.com/" in url:
                    parts = url.split("github.com/")
                    if len(parts) > 1:
                        repo_path = parts[1].strip("/")
                        # Extract owner/repo (first two components)
                        repo_parts = repo_path.split("/")
                        if len(repo_parts) >= 2:
                            repo_name = f"{repo_parts[0]}/{repo_parts[1]}"
                            target_repos.add(repo_name)

            print(f"  ✓ Loaded {csv_file.name}")

        except Exception as e:
            print(f"  ✗ Error reading {csv_file.name}: {e}")
            continue

    print(f"✓ Loaded {len(target_repos):,} unique target repositories\n")
    return target_repos


def load_config(config_path: str = "config.toml") -> Dict:
    """Load configuration from TOML file."""
    try:
        with open(config_path, "r") as f:
            config = toml.load(f)
        print(f"✓ Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Warning: {config_path} not found, using default weights")
        return {}


def get_weights_from_config(config: Dict) -> tuple:
    """Extract user_to_repo and repo_to_user weights from config."""
    # Default weights
    default_user_to_repo = {
        "COMMIT_CODE": 5,
        "PULL_REQUEST_OPENED": 20,
        "PULL_REQUEST_MERGED": 10,
        "STARRED": 5,
        "ISSUE_OPENED": 10,
        "FORKED": 1,
    }

    default_repo_to_user = {
        "COMMIT_CODE": 3,
        "PULL_REQUEST_OPENED": 5,
        "PULL_REQUEST_MERGED": 1,
    }

    # Get weights from config
    user_to_repo_weights = config.get("weights", {}).get("user_to_repo", {})
    repo_to_user_weights = config.get("weights", {}).get("repo_to_user", {})

    # Merge with defaults and normalize to uppercase
    user_to_repo = {**default_user_to_repo}
    if user_to_repo_weights:
        user_to_repo.update({k.upper(): v for k, v in user_to_repo_weights.items()})

    repo_to_user = {**default_repo_to_user}
    if repo_to_user_weights:
        repo_to_user.update({k.upper(): v for k, v in repo_to_user_weights.items()})

    return user_to_repo, repo_to_user


def is_bot(username: str, bot_keywords: list) -> bool:
    """Check if username matches bot patterns."""
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


def is_test_repo(repo_name: str, repo_exclude_keywords: list) -> bool:
    """Check if repository matches test/docs/example patterns."""
    if not repo_name:
        return False

    repo_lower = repo_name.lower()

    # Check for exclude keywords in repo name
    for keyword in repo_exclude_keywords:
        if keyword.lower() in repo_lower:
            return True

    return False


def process_and_append_cache_file(
    csv_file: Path,
    user_to_repo_weights: Dict[str, float],
    repo_to_user_weights: Dict[str, float],
    output_file: Path,
    target_repos: Set[str] = None,
    exclude_bots: bool = True,
    bot_keywords: list = None,
    exclude_test_repos: bool = True,
    repo_exclude_keywords: list = None,
    chunk_size: int = 100000,
):
    """
    Process a single cache file in chunks and append weighted results to output CSV.
    Does not aggregate - just applies weights and writes immediately.
    Filters out bot accounts if exclude_bots is True.
    Filters out test/docs/example repos if exclude_test_repos is True.
    Filters to only include repos in target_repos if target_repos is provided.
    """
    if bot_keywords is None:
        bot_keywords = []
    if repo_exclude_keywords is None:
        repo_exclude_keywords = []
    file_exists = output_file.exists()
    mode = "a" if file_exists else "w"
    header = not file_exists

    total_rows_written = 0

    # Process file in chunks
    for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
        # Normalize event_type to uppercase
        chunk["event_type"] = chunk["event_type"].str.upper()

        # Filter out invalid rows
        chunk = chunk.dropna(subset=["user", "repo", "event_type", "event_count"])
        chunk = chunk[chunk["event_count"] > 0]

        # Filter to only target repos if specified
        if target_repos:
            chunk = chunk[chunk["repo"].isin(target_repos)]

        # Filter out bots if enabled
        if exclude_bots and bot_keywords:
            chunk = chunk[~chunk["user"].apply(lambda x: is_bot(x, bot_keywords))]

        # Filter out test/docs/example repos if enabled
        if exclude_test_repos and repo_exclude_keywords:
            chunk = chunk[
                ~chunk["repo"].apply(lambda x: is_test_repo(x, repo_exclude_keywords))
            ]

        if chunk.empty:
            continue

        trust_records = []

        # Generate user-to-repo relationships
        for event_type, weight in user_to_repo_weights.items():
            if weight > 0:
                event_chunk = chunk[chunk["event_type"] == event_type].copy()
                if not event_chunk.empty:
                    event_chunk["v"] = event_chunk["event_count"] * weight
                    event_chunk = event_chunk.rename(columns={"user": "i", "repo": "j"})
                    trust_records.append(event_chunk[["i", "j", "v"]])

        # Generate repo-to-user relationships
        for event_type, weight in repo_to_user_weights.items():
            if weight > 0:
                event_chunk = chunk[chunk["event_type"] == event_type].copy()
                if not event_chunk.empty:
                    event_chunk["v"] = event_chunk["event_count"] * weight
                    event_chunk = event_chunk.rename(columns={"repo": "i", "user": "j"})
                    trust_records.append(event_chunk[["i", "j", "v"]])

        if trust_records:
            # Combine all records from this chunk
            combined_df = pd.concat(trust_records, ignore_index=True)

            # Append to output file
            combined_df.to_csv(output_file, mode=mode, header=header, index=False)

            total_rows_written += len(combined_df)

            # After first write, switch to append mode without header
            mode = "a"
            header = False

    return total_rows_written


def main():
    """Main execution function."""
    print("=" * 60)
    print("Generating Trust Relationships from Cache")
    print("=" * 60)
    print()

    # Load target repositories from ecosystems
    target_repos = load_target_repos()

    # Load configuration
    config = load_config()

    # Get weights from config
    user_to_repo_weights, repo_to_user_weights = get_weights_from_config(config)

    # Get filtering config
    filters = config.get("filters", {})
    exclude_bots = filters.get("exclude_bots", True)
    bot_keywords = filters.get("bot_keywords", [])
    exclude_test_repos = filters.get("exclude_test_repos", True)
    repo_exclude_keywords = filters.get("repo_exclude_keywords", [])

    print(f"Weights configuration:")
    print(f"  User-to-repo: {user_to_repo_weights}")
    print(f"  Repo-to-user: {repo_to_user_weights}")
    print()

    print(f"Filtering configuration:")
    print(f"  Exclude bots: {exclude_bots}")
    print(f"  Bot keywords: {len(bot_keywords)} keywords configured")
    print(f"  Exclude test/docs/example repos: {exclude_test_repos}")
    print(f"  Repo exclude keywords: {len(repo_exclude_keywords)} keywords configured")
    print()

    # Setup directories
    cache_dir = Path("cache")
    output_file = Path("trust/github.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing output file if it exists
    if output_file.exists():
        print(f"Removing existing output file: {output_file}")
        output_file.unlink()
        print()

    # Find all cache files
    csv_files = sorted(list(cache_dir.glob("*.csv")))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{cache_dir}'")

    print(f"Found {len(csv_files)} cache files to process")
    print(f"Processing each file with 1M row chunks...\n")

    # Process each file and append to output
    total_files = len(csv_files)
    grand_total_rows = 0

    for idx, csv_file in enumerate(csv_files, 1):
        print(
            f"[{idx}/{total_files}] Processing {csv_file.name}...",
            end=" ",
            flush=True,
        )

        try:
            # Process file and append to output
            rows_written = process_and_append_cache_file(
                csv_file,
                user_to_repo_weights,
                repo_to_user_weights,
                output_file,
                target_repos=target_repos if target_repos else None,
                exclude_bots=exclude_bots,
                bot_keywords=bot_keywords,
                exclude_test_repos=exclude_test_repos,
                repo_exclude_keywords=repo_exclude_keywords,
                chunk_size=1000000,
            )

            grand_total_rows += rows_written
            print(f"✓ ({rows_written:,} relationships written)")

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    # Print final summary
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)
    print(f"✓ Saved {grand_total_rows:,} trust relationships to {output_file}")
    print(f"  Total files processed: {total_files}")
    print(
        f"  Average relationships per file: {grand_total_rows // total_files:,} (approx)"
    )

    # Read a sample to show statistics (first 1M rows)
    print("\nReading sample for statistics (first 1M rows)...")
    try:
        sample_df = pd.read_csv(output_file, nrows=1000000)
        print(f"  Sample size: {len(sample_df):,} rows")
        print(f"  Sample trust value sum: {sample_df['v'].sum():,.2f}")
        print(f"  Sample average trust value: {sample_df['v'].mean():.2f}")
        print(f"  Sample min trust value: {sample_df['v'].min():.2f}")
        print(f"  Sample max trust value: {sample_df['v'].max():.2f}")

        print("\nSample relationships (first 10):")
        print("-" * 60)
        for idx, row in sample_df.head(10).iterrows():
            i_str = str(row["i"])[:30]
            j_str = str(row["j"])[:30]
            print(f"  {i_str:30s} -> {j_str:30s} : {row['v']:10.2f}")
    except Exception as e:
        print(f"  Could not read sample: {e}")

    print("\n" + "=" * 60)
    print("Trust generation complete!")
    print("=" * 60)
    print("\nNote: Output contains weighted relationships without aggregation.")
    print(
        "Duplicate (i,j) pairs exist and should be aggregated in post-processing if needed."
    )


if __name__ == "__main__":
    main()
