#!/usr/bin/env python3
"""
Database Import Script

This script imports data from CSV files into PostgreSQL database tables.

Tables populated:
- devrank.runs: Execution run metadata
- devrank.ecosystems: Repository URLs and sub-ecosystems
- devrank.scores: Score results for each run
- devrank.seeds: Seed data for each run

Note: For interactions import, use import_interactions.py

Usage:
    python import_to_db.py --all              # Import ecosystems and all runs
    python import_to_db.py --ecosystems       # Import ecosystems only
    python import_to_db.py --run <community>  # Create a new run and import scores/seeds

Requirements:
    - psycopg2 (install with: pip install psycopg2-binary)
    - toml (install with: pip install toml)
    - python-dotenv (install with: pip install python-dotenv)
"""

import argparse
import calendar
import csv
import os
import re
import sys
from pathlib import Path

import psycopg2
import toml
from dotenv import load_dotenv
from psycopg2.extras import execute_values

# Load environment variables from .env file
load_dotenv()


def load_config():
    """Load configuration from config.toml"""
    config_path = Path(__file__).parent / "config.toml"
    if not config_path.exists():
        print(f"❌ ERROR: config.toml not found at {config_path}")
        sys.exit(1)
    return toml.load(config_path)


def get_db_connection():
    """
    Get database connection from environment variables.

    Expected environment variables:
    - DATABASE_URL: Full connection string (e.g., postgresql://user:pass@host:port/dbname)
    """
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL environment variable is required")

    return psycopg2.connect(database_url)


def ensure_schema(conn):
    """Ensure the devrank schema exists"""
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS devrank")
    conn.commit()


def create_tables(conn):
    """Create all tables from schema files"""
    print("🔧 Creating tables from schema files...")

    schemas_dir = Path(__file__).parent / "schemas"

    # Order matters due to foreign key constraints
    schema_files = [
        "ecosystems.sql",
        "interactions.sql",
        "runs.sql",
        "scores.sql",
        "seeds.sql",
    ]

    for schema_file in schema_files:
        schema_path = schemas_dir / schema_file
        if schema_path.exists():
            print(f"  📄 Executing {schema_file}...")
            sql = schema_path.read_text()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                conn.commit()
            except Exception as e:
                print(f"    ❌ Error executing {schema_file}: {e}")
                conn.rollback()
        else:
            print(f"  ⚠️  Schema file not found: {schema_file}")

    # Verify tables were created
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'devrank'
        """)
        tables = [row[0] for row in cur.fetchall()]
        print(f"  📋 Tables in devrank schema: {tables}")

    print("  ✅ Tables created successfully")


def import_ecosystems(conn, ecosystems_dir):
    """
    Import ecosystem CSV files into devrank.ecosystems table.

    Args:
        conn: Database connection
        ecosystems_dir: Path to ecosystems directory
    """
    print("📦 Importing ecosystems...")

    ecosystems_path = Path(ecosystems_dir)
    csv_files = list(ecosystems_path.glob("*.csv"))

    if not csv_files:
        print(f"  ⚠️  No CSV files found in {ecosystems_dir}")
        return

    total_rows = 0

    with conn.cursor() as cur:
        for csv_file in csv_files:
            ecosystem_name = csv_file.stem
            print(f"  📄 Processing {ecosystem_name}...")

            try:
                with open(csv_file, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)

                    if "url" not in reader.fieldnames:
                        print(f"    ⚠️  Skipping {csv_file}: missing 'url' column")
                        continue

                    has_sub_ecosystems = "sub_ecosystems" in reader.fieldnames

                    rows = []
                    for row in reader:
                        url = row["url"]
                        sub_ecosystems = (
                            row.get("sub_ecosystems") if has_sub_ecosystems else None
                        )
                        if sub_ecosystems == "":
                            sub_ecosystems = None
                        rows.append((ecosystem_name, url, sub_ecosystems))

                # Insert with ON CONFLICT DO UPDATE
                execute_values(
                    cur,
                    """
                    INSERT INTO devrank.ecosystems (ecosystem_name, url, sub_ecosystems)
                    VALUES %s
                    ON CONFLICT (ecosystem_name, url) DO UPDATE SET
                        sub_ecosystems = EXCLUDED.sub_ecosystems,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    rows,
                    page_size=10000,
                )

                total_rows += len(rows)
                print(f"    ✅ Imported {len(rows)} rows")

            except Exception as e:
                print(f"    ❌ Error processing {csv_file}: {e}")

    conn.commit()
    print(f"  ✅ Total ecosystems imported: {total_rows}")


def calculate_days_back(cache_dir):
    """
    Calculate the total number of days covered by interaction files in cache.

    Args:
        cache_dir: Path to cache directory containing interaction files

    Returns:
        int: Total number of days across all months in cache
    """
    cache_path = Path(cache_dir)
    csv_files = list(cache_path.glob("interactions_*.csv"))

    if not csv_files:
        return 365  # Default fallback

    months = set()
    for csv_file in csv_files:
        match = re.match(r"interactions_(\d{2})_(\d{4})\.csv", csv_file.name)
        if match:
            month = int(match.group(1))
            year = int(match.group(2))
            months.add((year, month))

    if not months:
        return 365  # Default fallback

    total_days = 0
    for year, month in months:
        days_in_month = calendar.monthrange(year, month)[1]
        total_days += days_in_month

    return total_days


def get_next_run_id(conn, community_id):
    """
    Get the next run_id for a community.

    Args:
        conn: Database connection
        community_id: Community identifier

    Returns:
        int: The next run_id for this community (1 if no runs exist)
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(MAX(run_id), 0) + 1
            FROM devrank.runs
            WHERE community_id = %s
            """,
            (community_id,),
        )
        return cur.fetchone()[0]


def create_run(conn, community_id, ecosystems, days_back):
    """
    Create a new run entry in devrank.runs table.
    run_id is unique per community, not globally.

    Args:
        conn: Database connection
        community_id: Community identifier
        ecosystems: Comma-separated list of ecosystems (can be None)
        days_back: Number of days back for the run

    Returns:
        int: The created run_id (unique within this community)
    """
    run_id = get_next_run_id(conn, community_id)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO devrank.runs (run_id, community_id, ecosystems, days_back)
            VALUES (%s, %s, %s, %s)
            """,
            (run_id, community_id, ecosystems, days_back),
        )

    conn.commit()
    return run_id


def import_scores(conn, community_id, run_id, scores_file):
    """
    Import scores from a CSV file into devrank.scores table.

    Args:
        conn: Database connection
        community_id: The community_id to associate scores with
        run_id: The run_id to associate scores with
        scores_file: Path to the scores CSV file
    """
    print(f"📦 Importing scores from {scores_file}...")

    scores_path = Path(scores_file)
    if not scores_path.exists():
        print(f"  ❌ ERROR: File not found: {scores_file}")
        return 0

    try:
        with open(scores_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if "i" not in reader.fieldnames or "v" not in reader.fieldnames:
                print("  ❌ ERROR: Missing required columns 'i' and 'v'")
                return 0

            rows = []
            for row in reader:
                rows.append((community_id, run_id, row["i"], float(row["v"])))

            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO devrank.scores (community_id, run_id, user_id, value)
                    VALUES %s
                    ON CONFLICT (community_id, run_id, user_id) DO UPDATE SET
                        value = EXCLUDED.value
                    """,
                    rows,
                    page_size=10000,
                )

        conn.commit()
        print(f"  ✅ Imported {len(rows)} scores")
        return len(rows)

    except Exception as e:
        print(f"  ❌ Error importing scores: {e}")
        return 0


def import_seeds(conn, community_id, run_id, seed_file):
    """
    Import seeds from a CSV file into devrank.seeds table.

    Args:
        conn: Database connection
        community_id: The community_id to associate seeds with
        run_id: The run_id to associate seeds with
        seed_file: Path to the seed CSV file
    """
    print(f"📦 Importing seeds from {seed_file}...")

    seed_path = Path(seed_file)
    if not seed_path.exists():
        print(f"  ❌ ERROR: File not found: {seed_file}")
        return 0

    try:
        with open(seed_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if "i" not in reader.fieldnames or "v" not in reader.fieldnames:
                print("  ❌ ERROR: Missing required columns 'i' and 'v'")
                return 0

            rows = []
            for row in reader:
                rows.append((community_id, run_id, row["i"], float(row["v"])))

            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO devrank.seeds (community_id, run_id, user_id, value)
                    VALUES %s
                    ON CONFLICT (community_id, run_id, user_id) DO UPDATE SET
                        value = EXCLUDED.value
                    """,
                    rows,
                    page_size=10000,
                )

        conn.commit()
        print(f"  ✅ Imported {len(rows)} seeds")
        return len(rows)

    except Exception as e:
        print(f"  ❌ Error importing seeds: {e}")
        return 0


def import_run(conn, community_id, days_back, config):
    """
    Create a new run and import associated scores and seeds.

    Args:
        conn: Database connection
        community_id: Community identifier (e.g., 'bitcoin', 'eigenlayer')
        days_back: Number of days back for the run
        config: Configuration dictionary from config.toml
    """
    print(f"🚀 Creating run for community: {community_id}")

    # Get ecosystems mapping from config
    community_ecosystems = config.get("community_ecosystems", {})
    ecosystems = community_ecosystems.get(community_id)

    if ecosystems:
        print(f"  📋 Ecosystems: {ecosystems}")
    else:
        print(f"  ⚠️  No ecosystem mapping found for {community_id}")

    # Create the run
    run_id = create_run(conn, community_id, ecosystems, days_back)
    print(f"  ✅ Created run with ID: {run_id}")

    # Import scores
    base_dir = Path(__file__).parent
    scores_file = base_dir / "scores" / f"{community_id}.csv"
    seed_file = base_dir / "seed" / f"{community_id}.csv"

    scores_count = import_scores(conn, community_id, run_id, scores_file)
    seeds_count = import_seeds(conn, community_id, run_id, seed_file)

    print(f"  📊 Run summary: {scores_count} scores, {seeds_count} seeds")
    return run_id


def import_all_runs(conn, days_back, config):
    """
    Import all communities from the scores directory.

    Args:
        conn: Database connection
        days_back: Number of days back for all runs
        config: Configuration dictionary from config.toml
    """
    print("🚀 Importing all runs...")

    base_dir = Path(__file__).parent
    scores_dir = base_dir / "scores"

    csv_files = list(scores_dir.glob("*.csv"))

    if not csv_files:
        print(f"  ⚠️  No score files found in {scores_dir}")
        return

    for csv_file in csv_files:
        community_id = csv_file.stem
        print()
        import_run(conn, community_id, days_back, config)


def main():
    parser = argparse.ArgumentParser(
        description="Import data into DevRank PostgreSQL database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python import_to_db.py --all
  python import_to_db.py --ecosystems
  python import_to_db.py --run bitcoin

Environment variables:
  DATABASE_URL Full connection string (required)
               e.g., postgresql://user:pass@host:port/dbname
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Import all data (ecosystems, interactions, and all runs)",
    )

    parser.add_argument(
        "--ecosystems",
        action="store_true",
        help="Import ecosystem data only",
    )

    parser.add_argument(
        "--run",
        type=str,
        metavar="COMMUNITY_ID",
        help="Create a new run for the specified community and import its scores/seeds",
    )

    args = parser.parse_args()

    # Calculate days_back from cache files
    base_dir = Path(__file__).parent
    days_back = calculate_days_back(base_dir / "cache")
    print(f"📅 Calculated days_back from cache: {days_back} days")

    # Default to --all if no options specified
    if not any([args.all, args.ecosystems, args.run]):
        args.all = True

    # Load configuration
    config = load_config()

    # Connect to database
    print("🔌 Connecting to database...")
    try:
        conn = get_db_connection()
        print("  ✅ Connected successfully")
    except Exception as e:
        print(f"  ❌ ERROR: Failed to connect to database: {e}")
        sys.exit(1)

    try:
        # Ensure schema exists and create tables
        ensure_schema(conn)
        create_tables(conn)

        base_dir = Path(__file__).parent

        if args.all:
            # Import everything
            import_ecosystems(conn, base_dir / "ecosystems")
            print()
            import_all_runs(conn, days_back, config)
        else:
            if args.ecosystems:
                import_ecosystems(conn, base_dir / "ecosystems")

            if args.run:
                import_run(conn, args.run, days_back, config)

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
