#!/usr/bin/env python3
"""
Import Interactions Script

This script imports interaction data from cache CSV files into devrank.interactions table.
It filters interactions to only include repos that exist in the ecosystems directory.

Features:
- Loads ecosystem repos for filtering
- Uses TRUNCATE and COPY for fast bulk loading
- Supports checkpointing to resume interrupted imports
- Creates indexes after all data is loaded

Usage:
    python import_interactions.py
    python import_interactions.py --no-index    # Skip index creation
    python import_interactions.py --reset       # Clear checkpoint and start fresh

Requirements:
    - psycopg2 (install with: pip install psycopg2-binary)
    - python-dotenv (install with: pip install python-dotenv)
"""

import argparse
import csv
import io
import os
import re
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    """Get database connection from DATABASE_URL environment variable."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is required")
    return psycopg2.connect(database_url)


def load_ecosystem_repos(ecosystems_dir):
    """
    Load all repos from ecosystem files.

    Args:
        ecosystems_dir: Path to ecosystems directory

    Returns:
        set: Set of repo names (owner/repo format)
    """
    repos = set()
    ecosystems_path = Path(ecosystems_dir)
    csv_files = list(ecosystems_path.glob("*.csv"))

    for csv_file in csv_files:
        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "url" not in reader.fieldnames:
                continue
            for row in reader:
                url = row["url"]
                # Extract owner/repo from https://github.com/owner/repo
                if "github.com/" in url:
                    repo = url.split("github.com/")[-1].rstrip("/")
                    repos.add(repo)

    return repos


def load_checkpoint(checkpoint_file):
    """Load completed files from checkpoint"""
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_checkpoint(checkpoint_file, filename):
    """Append completed file to checkpoint"""
    with open(checkpoint_file, "a") as f:
        f.write(f"{filename}\n")


def create_table(conn):
    """Create devrank.interactions table if it doesn't exist."""
    print("🔧 Creating devrank.interactions table...")

    sql = """
    CREATE TABLE IF NOT EXISTS devrank.interactions (
        id SERIAL PRIMARY KEY,
        user_login VARCHAR(255) NOT NULL,
        repo VARCHAR(512) NOT NULL,
        event_type VARCHAR(50) NOT NULL,
        event_count INTEGER NOT NULL DEFAULT 0,
        year INTEGER NOT NULL,
        month INTEGER NOT NULL
    )
    """

    try:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS devrank")
            cur.execute(sql)
        conn.commit()
        print("  ✅ Table created")
    except Exception as e:
        print(f"  ❌ Error creating table: {e}")
        conn.rollback()
        sys.exit(1)


def create_indexes(conn):
    """Create indexes on devrank.interactions table after bulk loading."""
    print("📊 Creating indexes on devrank.interactions...")

    indexes = [
        (
            "idx_interactions_user_login",
            "CREATE INDEX IF NOT EXISTS idx_interactions_user_login ON devrank.interactions(user_login)",
        ),
        (
            "idx_interactions_repo",
            "CREATE INDEX IF NOT EXISTS idx_interactions_repo ON devrank.interactions(repo)",
        ),
        (
            "idx_interactions_event_type",
            "CREATE INDEX IF NOT EXISTS idx_interactions_event_type ON devrank.interactions(event_type)",
        ),
        (
            "idx_interactions_year_month",
            "CREATE INDEX IF NOT EXISTS idx_interactions_year_month ON devrank.interactions(year, month)",
        ),
        (
            "idx_interactions_user_repo",
            "CREATE INDEX IF NOT EXISTS idx_interactions_user_repo ON devrank.interactions(user_login, repo)",
        ),
        (
            "idx_interactions_unique",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_interactions_unique ON devrank.interactions(user_login, repo, event_type, year, month)",
        ),
    ]

    for name, sql in indexes:
        print(f"  🔨 Creating {name}...")
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            print(f"    ✅ Created {name}")
        except Exception as e:
            print(f"    ❌ Error creating {name}: {e}")
            conn.rollback()

    print("  ✅ Indexing complete!")


def import_interactions(conn, cache_dir, ecosystems_dir, reset=False):
    """
    Import interaction CSV files into devrank.interactions table.
    Uses TRUNCATE and COPY for fast bulk loading.
    Filters to only include repos from ecosystems.
    Supports checkpointing to resume interrupted imports.

    Args:
        conn: Database connection
        cache_dir: Path to cache directory containing interaction files
        ecosystems_dir: Path to ecosystems directory for filtering repos
        reset: If True, clear checkpoint and start fresh
    """
    print("📦 Importing interactions...")

    # Load ecosystem repos for filtering
    print("  📂 Loading ecosystem repos...")
    ecosystem_repos = load_ecosystem_repos(ecosystems_dir)
    print(f"  📋 Found {len(ecosystem_repos)} repos in ecosystems")

    cache_path = Path(cache_dir)
    csv_files = list(cache_path.glob("interactions_*.csv"))

    if not csv_files:
        print(f"  ⚠️  No interaction files found in {cache_dir}")
        return

    # Checkpoint file to track completed imports
    checkpoint_file = cache_path / ".interactions_checkpoint"

    if reset and checkpoint_file.exists():
        checkpoint_file.unlink()
        print("  🗑️  Checkpoint cleared")

    completed_files = load_checkpoint(checkpoint_file)

    if completed_files:
        print(
            f"  📋 Resuming from checkpoint ({len(completed_files)} files already done)"
        )
    else:
        # Only truncate if starting fresh (no checkpoint)
        try:
            with conn.cursor() as cur:
                print("  🗑️  Truncating devrank.interactions...")
                cur.execute("TRUNCATE TABLE devrank.interactions RESTART IDENTITY")
            conn.commit()
        except Exception as e:
            print(f"  ❌ Error truncating table: {e}")
            conn.rollback()
            return

    total_rows = 0

    for csv_file in sorted(csv_files):
        # Skip already completed files
        if csv_file.name in completed_files:
            print(f"  ⏭️  Skipping {csv_file.name} (already imported)")
            continue

        # Extract month and year from filename (e.g., interactions_01_2024.csv)
        match = re.match(r"interactions_(\d{2})_(\d{4})\.csv", csv_file.name)
        if not match:
            print(f"  ⚠️  Skipping {csv_file}: unexpected filename format")
            continue

        month = int(match.group(1))
        year = int(match.group(2))

        print(f"  📄 Processing {csv_file.name} (month={month}, year={year})...")

        try:
            with open(csv_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                required_cols = ["user", "repo", "event_type", "event_count"]
                if not all(col in reader.fieldnames for col in required_cols):
                    print(f"    ⚠️  Skipping {csv_file}: missing required columns")
                    continue

                # Build CSV data in memory for COPY (filtered by ecosystem repos)
                buffer = io.StringIO()
                row_count = 0
                skipped_count = 0
                for row in reader:
                    repo = row["repo"]
                    # Only include repos that are in ecosystems
                    if repo in ecosystem_repos:
                        buffer.write(
                            f"{row['user']}\t{repo}\t{row['event_type']}\t{row['event_count']}\t{year}\t{month}\n"
                        )
                        row_count += 1
                    else:
                        skipped_count += 1

                buffer.seek(0)

                # Use COPY for fast bulk insert with fresh cursor
                with conn.cursor() as cur:
                    cur.copy_expert(
                        "COPY devrank.interactions (user_login, repo, event_type, event_count, year, month) FROM STDIN",
                        buffer,
                    )

            conn.commit()
            total_rows += row_count
            print(
                f"    ✅ Imported {row_count:,} rows (skipped {skipped_count:,} non-ecosystem repos)"
            )

            # Save checkpoint after successful import
            save_checkpoint(checkpoint_file, csv_file.name)

        except Exception as e:
            print(f"    ❌ Error processing {csv_file}: {e}")
            conn.rollback()

    # Remove checkpoint file if all files completed successfully
    if checkpoint_file.exists():
        remaining = set(f.name for f in csv_files) - load_checkpoint(checkpoint_file)
        if not remaining:
            checkpoint_file.unlink()
            print("  🧹 Checkpoint file removed (all files imported)")

    print(f"  ✅ Total interactions imported: {total_rows:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Import interaction data into devrank.interactions table"
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Skip index creation after import",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear checkpoint and start fresh (will truncate table)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="Directory containing interaction CSV files (default: cache)",
    )
    parser.add_argument(
        "--ecosystems-dir",
        type=str,
        default="ecosystems",
        help="Directory containing ecosystem CSV files (default: ecosystems)",
    )

    args = parser.parse_args()

    # Connect to database
    print("🔌 Connecting to database...")
    try:
        conn = get_db_connection()
        print("  ✅ Connected successfully")
    except Exception as e:
        print(f"  ❌ ERROR: Failed to connect to database: {e}")
        sys.exit(1)

    try:
        # Create table if needed
        create_table(conn)

        # Import interactions
        import_interactions(
            conn,
            args.cache_dir,
            args.ecosystems_dir,
            reset=args.reset,
        )

        # Create indexes unless skipped
        if not args.no_index:
            create_indexes(conn)

        print("\n✅ Import complete!")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
