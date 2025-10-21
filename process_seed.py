#!/usr/bin/env python3
"""
General purpose script to process seed score CSV files.
Loads a CSV file, identifies tier sections, assigns weighted scores, and creates a backup.
"""

import pandas as pd
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import shutil

def create_backup(csv_path):
    """
    Create a backup of the original CSV file with timestamp.

    Args:
        csv_path (Path): Path to the original CSV file

    Returns:
        Path: Path to the backup file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = csv_path.parent / f"{csv_path.stem}_backup_{timestamp}{csv_path.suffix}"

    try:
        shutil.copy2(csv_path, backup_path)
        print(f"🔄 Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"❌ ERROR: Failed to create backup: {e}")
        sys.exit(1)

def process_seed_csv(csv_path, tier_weights=None):
    """
    Process a seed scores CSV file to assign tier-based scores.

    Args:
        csv_path (str or Path): Path to the CSV file to process
        tier_weights (list): List of weights for each tier (default: [0.6, 0.2, 0.2])

    Expected format:
    - Tier 1 repositories (default 60% weight)
    - Empty line
    - Tier 2 repositories (default 20% weight)
    - Empty line
    - Tier 3 repositories (default 20% weight)
    """

    if tier_weights is None:
        tier_weights = [0.6, 0.2, 0.2]

    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"❌ ERROR: File {csv_path} not found!")
        sys.exit(1)

    print(f"📂 Loading {csv_path}...")

    # Create backup before processing
    backup_path = create_backup(csv_path)

    try:
        # First, read the raw file to map original positions to DataFrame indices
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Map original line numbers to DataFrame row indices
        df_row_to_original_line = {}
        original_line_to_df_row = {}
        df_row = 0

        for line_num, line in enumerate(lines):
            if line_num == 0:  # Skip header
                continue
            if line.strip() != '':  # Non-empty line becomes a DataFrame row
                df_row_to_original_line[df_row] = line_num
                original_line_to_df_row[line_num] = df_row
                df_row += 1

        # Find empty lines in original file
        empty_lines = []
        for line_num, line in enumerate(lines):
            if line_num > 0 and line.strip() == '':  # Skip header, find empty lines
                empty_lines.append(line_num)

        # Read the CSV file
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} rows")

        # Convert empty line positions to DataFrame indices for tier boundaries
        tier_boundaries = []
        if len(empty_lines) >= len(tier_weights) - 1:
            print(f"🔍 Found {len(empty_lines)} separator rows at original lines: {empty_lines}")

            # First tier: from start to first empty line
            first_empty = empty_lines[0]
            tier1_end = 0
            for df_idx, orig_line in df_row_to_original_line.items():
                if orig_line >= first_empty:
                    break
                tier1_end = df_idx + 1
            tier_boundaries.append((0, tier1_end))

            # Middle tiers: between consecutive empty lines
            for i in range(len(tier_weights) - 2):
                if i + 1 < len(empty_lines):
                    start_line = empty_lines[i]
                    end_line = empty_lines[i + 1]

                    tier_start = None
                    tier_end = None

                    for df_idx, orig_line in df_row_to_original_line.items():
                        if tier_start is None and orig_line > start_line:
                            tier_start = df_idx
                        if orig_line >= end_line:
                            tier_end = df_idx
                            break

                    if tier_start is not None:
                        if tier_end is None:
                            tier_end = len(df)
                        tier_boundaries.append((tier_start, tier_end))

            # Last tier: from last empty line to end
            last_empty = empty_lines[-1]
            last_tier_start = None
            for df_idx, orig_line in df_row_to_original_line.items():
                if orig_line > last_empty:
                    last_tier_start = df_idx
                    break
            if last_tier_start is not None:
                tier_boundaries.append((last_tier_start, len(df)))

        elif len(empty_lines) > 0:
            print(f"🔍 Found {len(empty_lines)} separator rows - using equal division for remaining tiers")
            # Some separators, but not enough - distribute remaining rows
            first_empty = empty_lines[0]
            tier1_end = 0
            for df_idx, orig_line in df_row_to_original_line.items():
                if orig_line >= first_empty:
                    break
                tier1_end = df_idx + 1
            tier_boundaries.append((0, tier1_end))

            remaining_start = tier1_end
            remaining_rows = len(df) - remaining_start
            remaining_tiers = len(tier_weights) - 1
            rows_per_tier = remaining_rows // remaining_tiers

            for i in range(remaining_tiers - 1):
                start_idx = remaining_start + i * rows_per_tier
                end_idx = remaining_start + (i + 1) * rows_per_tier
                tier_boundaries.append((start_idx, end_idx))

            # Last tier gets remaining rows
            tier_boundaries.append((remaining_start + (remaining_tiers - 1) * rows_per_tier, len(df)))

        else:
            print(f"🔍 Found 0 separator rows - using equal division")
            # No separators, divide equally into tiers
            rows_per_tier = len(df) // len(tier_weights)

            for i in range(len(tier_weights) - 1):
                start_idx = i * rows_per_tier
                end_idx = (i + 1) * rows_per_tier
                tier_boundaries.append((start_idx, end_idx))

            # Last tier gets remaining rows
            tier_boundaries.append(((len(tier_weights) - 1) * rows_per_tier, len(df)))

        # Initialize or reset the 'v' column (seed score column)
        if 'v' not in df.columns:
            df['v'] = 0.0
        else:
            df['v'] = 0.0

        # Count repositories in each tier and assign scores
        tier_info = []

        for tier_idx, (start, end) in enumerate(tier_boundaries):
            if tier_idx >= len(tier_weights):
                break

            # Count actual repositories (non-empty rows) in this tier
            tier_repos = 0
            for i in range(start, end):
                if i < len(df) and not df.iloc[i].isnull().all():
                    tier_repos += 1

            tier_weight = tier_weights[tier_idx]
            tier_score = tier_weight / tier_repos if tier_repos > 0 else 0

            tier_info.append({
                'tier': tier_idx + 1,
                'repos': tier_repos,
                'weight': tier_weight,
                'score_per_repo': tier_score,
                'start': start,
                'end': end
            })

        # Display tier distribution
        print(f"📊 Tier distribution:")
        for info in tier_info:
            print(f"   Tier {info['tier']}: {info['repos']} repositories ({info['weight']*100:.1f}% weight)")

        print(f"💯 Score per repository:")
        for info in tier_info:
            print(f"   Tier {info['tier']}: {info['score_per_repo']:.6f}")

        # Assign scores to each tier
        for info in tier_info:
            for i in range(info['start'], info['end']):
                if i < len(df) and not df.iloc[i].isnull().all():
                    df.iloc[i, df.columns.get_loc('v')] = info['score_per_repo']

        # Verify total score
        total_score = df['v'].sum()
        expected_total = sum(tier_weights)
        print(f"🎯 Total score: {total_score:.6f} (expected: {expected_total:.6f})")

        # Save back to the original file
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved updated file to {csv_path}")

        # Show sample of results
        print(f"\n📋 Sample results:")
        non_zero_scores = df[df['v'] > 0].head(10)
        if len(non_zero_scores) > 0:
            first_col = df.columns[0]
            for i, row in non_zero_scores.iterrows():
                print(f"   {row[first_col]} -> {row['v']:.6f}")
        else:
            print("   No non-zero scores found")

    except Exception as e:
        print(f"❌ ERROR: Failed to process file: {e}")
        print(f"🔄 Backup file preserved at: {backup_path}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Process seed score CSV files with tier-based weighting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_seed.py seed_scores.csv
  python process_seed.py /path/to/scores.csv --weights 0.5 0.3 0.2
  python process_seed.py data.csv --weights 0.4 0.4 0.2

The script expects CSV files with tier sections separated by empty rows.
A backup file will be created automatically before processing.
        """)

    parser.add_argument("csv_path",
                       help="Path to the CSV file to process")

    parser.add_argument("--weights", "-w",
                       nargs="+",
                       type=float,
                       default=[0.6, 0.2, 0.2],
                       help="Weights for each tier (default: 0.6 0.2 0.2)")

    args = parser.parse_args()

    # Validate weights sum to 1.0 (with small tolerance for float precision)
    weight_sum = sum(args.weights)
    if abs(weight_sum - 1.0) > 0.001:
        print(f"❌ ERROR: Weights must sum to 1.0, got {weight_sum:.6f}")
        print(f"Current weights: {args.weights}")
        sys.exit(1)

    print("🚀 Processing seed scores CSV...")
    print(f"📁 File: {args.csv_path}")
    print(f"⚖️  Weights: {args.weights}")

    process_seed_csv(args.csv_path, args.weights)
    print("✅ Processing complete!")

if __name__ == "__main__":
    main()
