#!/usr/bin/env python3
"""
Score Processing Script

This script processes score files from the scores/ directory by:
1. Loading all CSV score files
2. Splitting peers into organizations (containing '/') and developers (no '/')
3. Applying a single transformation (log by default, or sqrt/quantile with flags)
4. Normalizing scores within each group so all scores sum to 1
5. Saving results to output/ directory

Transformations available (choose one):
- Logarithmic (default): log transformation (first scaled to 10-100 range) to linearize exponential data
- Square Root (--sqrt flag): sqrt transformation for gentle compression of higher values
- Quantile (--quantile flag): uniform distribution preserving rank order

All transformations output scores in the 0-1000 range, where:
- Scores are normalized within each group (orgs and devs separately)
- 0 represents the minimum score in the group
- 1000 represents the maximum score in the group

Usage:
    python3 process_scores.py            # Log transformation (default)
    python3 process_scores.py --sqrt     # Sqrt transformation
    python3 process_scores.py --quantile # Quantile transformation

Requirements:
    - pandas (install with: pip install pandas)
    - numpy (install with: pip install numpy)
    - scipy (for quantile transformation, only needed if --quantile is used)
    - CSV files in scores/ directory with columns 'i' (identifier) and 'v' (score)

Output:
    - Creates output/ directory if it doesn't exist
    - For each input file (e.g., bitcoin.csv), creates:
      - {filename}_orgs.csv: Organizations with selected transformation
      - {filename}_devs.csv: Developers with selected transformation
    - All scores are in the 0-1000 range, normalized within each group (orgs/devs separately)
"""

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def normalize_scores(df):
    """
    Normalize scores so that all scores sum to 1, then map to 0-1000 range

    Args:
        df (pandas.DataFrame): DataFrame with 'v' column containing scores

    Returns:
        pandas.DataFrame: DataFrame with normalized scores mapped to 0-1000 range
    """
    if len(df) == 0:
        return df

    df_normalized = df.copy()
    total_score = df["v"].sum()

    # Avoid division by zero if all scores are zero
    if total_score == 0:
        df_normalized["v"] = 1.0 / len(df)  # Equal distribution
    else:
        df_normalized["v"] = df["v"] / total_score

    # Map to 0-1000 range
    df_normalized["v"] = df_normalized["v"] * 1000

    # Round to 2 decimal places
    df_normalized["v"] = df_normalized["v"].round(2)

    return df_normalized


def apply_sqrt_transformation(df):
    """
    Apply square root transformation to scores.

    Scores are normalized to 0-1, sqrt is applied, then mapped to 0-1000 range.
    """
    if len(df) == 0:
        return df

    df_transformed = df.copy()

    # Apply sqrt transformation
    df_transformed["v"] = np.sqrt(df["v"])

    # Normalize to 0-1 range
    min_val = df_transformed["v"].min()
    max_val = df_transformed["v"].max()
    if max_val != min_val:
        df_transformed["v"] = (df_transformed["v"] - min_val) / (max_val - min_val)
    else:
        df_transformed["v"] = 1.0 / len(df)

    # Map to 0-1000 range
    df_transformed["v"] = df_transformed["v"] * 1000

    # Round to 2 decimal places
    df_transformed["v"] = df_transformed["v"].round(2)

    return df_transformed


def apply_log_transformation(df):
    """
    Apply logarithmic transformation to scores.

    Process:
    1. Normalize scores to 0-1 range
    2. Scale to 100-1000 range (wider range for better differentiation)
    3. Apply natural logarithm
    4. Re-normalize log values to 0-1
    5. Map to 0-1000 output range
    """
    if len(df) == 0:
        return df

    df_transformed = df.copy()

    # First normalize to 0-1 range
    min_val = df["v"].min()
    max_val = df["v"].max()
    if max_val != min_val:
        df_transformed["v"] = (df["v"] - min_val) / (max_val - min_val)
    else:
        df_transformed["v"] = 1.0 / len(df)

    # Map to 100-1000 range for better log distribution
    df_transformed["v"] = df_transformed["v"] * 900 + 100

    # Apply log transformation
    df_transformed["v"] = np.log(df_transformed["v"])

    # Normalize back to 0-1 range
    min_log = df_transformed["v"].min()
    max_log = df_transformed["v"].max()
    if max_log != min_log:
        df_transformed["v"] = (df_transformed["v"] - min_log) / (max_log - min_log)
    else:
        df_transformed["v"] = 1.0 / len(df)

    # Map to 0-1000 range
    df_transformed["v"] = df_transformed["v"] * 1000

    # Round to 2 decimal places
    df_transformed["v"] = df_transformed["v"].round(2)

    return df_transformed


def apply_quantile_transformation(df):
    """
    Apply quantile-based uniform distribution transformation.

    Maps ranks to uniform distribution in 0-1000 range, preserving rank order.
    """
    if len(df) == 0:
        return df

    # Import scipy locally when needed
    from scipy import stats

    df_transformed = df.copy()

    # Use scipy for quantile transformation
    df_transformed["v"] = stats.rankdata(df["v"]) / len(df["v"])

    # Map to 0-1000 range
    df_transformed["v"] = df_transformed["v"] * 1000

    # Round to 2 decimal places
    df_transformed["v"] = df_transformed["v"].round(2)

    return df_transformed


def split_and_process_scores(input_file, output_dir, transformation="log"):
    """
    Process a single score file by splitting into orgs and devs, applying transformation, and saving

    Args:
        input_file (str): Path to input CSV file
        output_dir (str): Directory to save processed files
        transformation (str): Which transformation to apply: "log", "sqrt", or "quantile"
    """
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Split into organizations (containing '/') and developers (no '/')
    orgs_df = df[df["i"].str.contains("/", na=False)].copy()
    devs_df = df[~df["i"].str.contains("/", na=False)].copy()

    # Select transformation function
    transform_funcs = {
        "log": apply_log_transformation,
        "sqrt": apply_sqrt_transformation,
        "quantile": apply_quantile_transformation,
    }

    transform_func = transform_funcs[transformation]

    base_name = Path(input_file).stem
    print(f"Processing {input_file}:")

    # Apply transformation to both groups
    if len(orgs_df) > 0:
        orgs_transformed = transform_func(orgs_df)
    else:
        orgs_transformed = orgs_df.copy()

    if len(devs_df) > 0:
        devs_transformed = transform_func(devs_df)
    else:
        devs_transformed = devs_df.copy()

    # Generate output file names (without transformation suffix)
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

    print(f"  - {transformation.capitalize()} transformation:")
    print(f"    Organizations: {len(orgs_transformed)} entries -> {orgs_output}")
    print(f"    Score range: {orgs_min:.2f} - {orgs_max:.2f}")
    print(f"    Developers: {len(devs_transformed)} entries -> {devs_output}")
    print(f"    Score range: {devs_min:.2f} - {devs_max:.2f}")


def main():
    """
    Main function to process all score files
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process score files with a single transformation (log by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_scores.py            # Log transformation (default)
  python process_scores.py --sqrt     # Sqrt transformation
  python process_scores.py --quantile # Quantile transformation

Only one transformation can be applied at a time.
        """,
    )

    parser.add_argument(
        "--sqrt",
        action="store_true",
        help="Apply square root transformation (instead of log)",
    )

    parser.add_argument(
        "--quantile",
        action="store_true",
        help="Apply quantile transformation (instead of log)",
    )

    args = parser.parse_args()

    # Check that only one transformation is specified
    if args.sqrt and args.quantile:
        parser.error(
            "Cannot specify both --sqrt and --quantile. Choose only one transformation."
        )

    # Determine which transformation to use
    if args.sqrt:
        transformation = "sqrt"
    elif args.quantile:
        transformation = "quantile"
    else:
        transformation = "log"

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

    # Show which transformation will be applied
    print(f"Found {len(csv_files)} score files to process...")
    print(f"Transformation: {transformation}")
    print()

    # Process each CSV file
    for csv_file in csv_files:
        try:
            split_and_process_scores(
                str(csv_file),
                output_dir,
                transformation=transformation,
            )
            print()
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            print()

    print("Processing complete!")


if __name__ == "__main__":
    main()
