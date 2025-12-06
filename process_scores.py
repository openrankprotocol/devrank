#!/usr/bin/env python3
"""
Score Processing Script

This script processes score files from the scores/ directory by:
1. Loading all CSV score files
2. Splitting peers into organizations (containing '/') and developers (no '/')
3. Optionally applying log transformation (with --log flag)
4. Saving results to output/ directory

Usage:
    python3 process_scores.py          # Split only, no transformation
    python3 process_scores.py --log    # Split and apply log transformation

Requirements:
    - pandas (install with: pip install pandas)
    - numpy (install with: pip install numpy)
    - CSV files in scores/ directory with columns 'i' (identifier) and 'v' (score)

Output:
    - Creates output/ directory if it doesn't exist
    - For each input file (e.g., bitcoin.csv), creates:
      - {filename}_orgs.csv: Organizations
      - {filename}_devs.csv: Developers
    - With --log flag, scores are mapped to 0.0-1.0 range
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def apply_log_transformation(df):
    """
    Apply logarithmic transformation to scores.

    Process:
    1. Apply natural logarithm to values
    2. Map to 0.0-1.0 range
    """
    if len(df) == 0:
        return df

    df_transformed = df.copy()

    # Apply log transformation (add small epsilon to avoid log(0))
    df_transformed["v"] = np.log(df["v"] + 1e-10)

    # Map to 0.0-1.0 range
    min_log = df_transformed["v"].min()
    max_log = df_transformed["v"].max()
    if max_log != min_log:
        df_transformed["v"] = (df_transformed["v"] - min_log) / (max_log - min_log)
    else:
        df_transformed["v"] = 1.0 / len(df)

    # Round to 6 decimal places
    df_transformed["v"] = df_transformed["v"].round(6)

    return df_transformed


def split_and_process_scores(input_file, output_dir, apply_log=False):
    """
    Process a single score file by splitting into orgs and devs, optionally applying transformation, and saving

    Args:
        input_file (str): Path to input CSV file
        output_dir (str): Directory to save processed files
        apply_log (bool): Whether to apply log transformation
    """
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Split into organizations (containing '/') and developers (no '/')
    orgs_df = df[df["i"].str.contains("/", na=False)].copy()
    devs_df = df[~df["i"].str.contains("/", na=False)].copy()

    base_name = Path(input_file).stem
    print(f"Processing {input_file}:")

    # Apply transformation to both groups if requested
    if apply_log:
        if len(orgs_df) > 0:
            orgs_transformed = apply_log_transformation(orgs_df)
        else:
            orgs_transformed = orgs_df.copy()

        if len(devs_df) > 0:
            devs_transformed = apply_log_transformation(devs_df)
        else:
            devs_transformed = devs_df.copy()
    else:
        orgs_transformed = orgs_df.copy()
        devs_transformed = devs_df.copy()

    # Generate output file names
    orgs_output = os.path.join(output_dir, f"{base_name}_orgs.csv")
    devs_output = os.path.join(output_dir, f"{base_name}_devs.csv")

    # Save the processed files
    orgs_transformed.to_csv(orgs_output, index=False)
    devs_transformed.to_csv(devs_output, index=False)

    # Show score ranges
    orgs_min = orgs_transformed["v"].min() if len(orgs_transformed) > 0 else 0
    orgs_max = orgs_transformed["v"].max() if len(orgs_transformed) > 0 else 0
    devs_min = devs_transformed["v"].min() if len(devs_transformed) > 0 else 0
    devs_max = devs_transformed["v"].max() if len(devs_transformed) > 0 else 0

    print(f"  Organizations: {len(orgs_transformed)} entries -> {orgs_output}")
    print(f"    Score range: {orgs_min:.6f} - {orgs_max:.6f}")
    print(f"  Developers: {len(devs_transformed)} entries -> {devs_output}")
    print(f"    Score range: {devs_min:.6f} - {devs_max:.6f}")


def main():
    """
    Main function to process all score files
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process score files by splitting into orgs and devs"
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Apply log transformation and map to 0.0-1.0 range",
    )
    args = parser.parse_args()

    # Define directories
    scores_dir = "scores"
    output_dir = "output"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all CSV files in the scores directory
    scores_path = Path(scores_dir)
    csv_files = list(scores_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {scores_dir} directory")
        return

    print(f"Found {len(csv_files)} score files to process...")
    if args.log:
        print("Applying log transformation")
    print()

    # Process each CSV file
    for csv_file in csv_files:
        try:
            split_and_process_scores(str(csv_file), output_dir, apply_log=args.log)
            print()
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            print()

    print("Processing complete!")


if __name__ == "__main__":
    main()
