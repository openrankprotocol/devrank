#!/usr/bin/env python3
"""
Score Processing Script

This script processes score files from the scores/ directory by:
1. Loading all CSV score files
2. Splitting peers into organizations (containing '/') and developers (no '/')
3. Applying transformations to make exponential distributions more linear
4. Normalizing scores within each group so all scores sum to 1
5. Saving results to output/ directory with transformation suffixes

Transformations available:
- Logarithmic: log transformation to linearize exponential data
- Quantile: uniform distribution preserving rank order

Usage:
    python3 process_scores.py

Requirements:
    - pandas (install with: pip install pandas)
    - numpy (install with: pip install numpy)
    - scipy (optional, for quantile transformation - will use fallback if not available)
    - CSV files in scores/ directory with columns 'i' (identifier) and 'v' (score)

Output:
    - Creates output/ directory if it doesn't exist
    - For each input file (e.g., ai.csv), creates:
      - {filename}_orgs_log.csv: Organizations with logarithmic transformation
      - {filename}_devs_log.csv: Developers with logarithmic transformation

      - {filename}_orgs_quantile.csv: Organizations with quantile transformation
      - {filename}_devs_quantile.csv: Developers with quantile transformation
    - Scores are normalized within each group (sum to 1) and sorted by score (descending)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Try to import scipy, use fallback if not available
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using fallback for quantile transformation")


def normalize_scores(df):
    """
    Normalize scores so that all scores sum to 1, then map to 100-1000 range

    Args:
        df (pandas.DataFrame): DataFrame with 'v' column containing scores

    Returns:
        pandas.DataFrame: DataFrame with normalized scores mapped to 100-1000 range
    """
    if len(df) == 0:
        return df

    df_normalized = df.copy()
    total_score = df['v'].sum()

    # Avoid division by zero if all scores are zero
    if total_score == 0:
        df_normalized['v'] = 1.0 / len(df)  # Equal distribution
    else:
        df_normalized['v'] = df['v'] / total_score

    # Map to 100-1000 range
    df_normalized['v'] = df_normalized['v'] * 900 + 100

    # Round to 2 decimal places
    df_normalized['v'] = df_normalized['v'].round(2)

    return df_normalized


def apply_log_transformation(df):
    """Apply logarithmic transformation to scores"""
    if len(df) == 0:
        return df

    df_transformed = df.copy()
    min_score = df['v'].min()
    epsilon = min_score * 0.1 if min_score > 0 else 0.000001

    # Apply log transformation
    df_transformed['v'] = np.log(df['v'] + epsilon)

    # Normalize to 0-1 range
    min_val = df_transformed['v'].min()
    max_val = df_transformed['v'].max()
    if max_val != min_val:
        df_transformed['v'] = (df_transformed['v'] - min_val) / (max_val - min_val)
    else:
        df_transformed['v'] = 1.0 / len(df)

    # Map to 100-1000 range
    df_transformed['v'] = df_transformed['v'] * 900 + 100

    # Round to 2 decimal places
    df_transformed['v'] = df_transformed['v'].round(2)

    return df_transformed





def apply_quantile_transformation(df):
    """Apply quantile-based uniform distribution transformation"""
    if len(df) == 0:
        return df

    df_transformed = df.copy()

    if SCIPY_AVAILABLE:
        # Use scipy for optimal performance
        df_transformed['v'] = stats.rankdata(df['v']) / len(df['v'])
    else:
        # Fallback implementation: create rank mapping preserving original order
        sorted_df = df.sort_values('v', ascending=True, kind='stable')
        ranks = np.arange(1, len(sorted_df) + 1) / len(sorted_df)
        rank_mapping = dict(zip(sorted_df.index, ranks))
        df_transformed['v'] = df_transformed.index.map(rank_mapping)

    # Map to 100-1000 range
    df_transformed['v'] = df_transformed['v'] * 900 + 100

    # Round to 2 decimal places
    df_transformed['v'] = df_transformed['v'].round(2)

    return df_transformed


def split_and_process_scores(input_file, output_dir):
    """
    Process a single score file by splitting into orgs and devs, applying transformations, and saving

    Args:
        input_file (str): Path to input CSV file
        output_dir (str): Directory to save processed files
    """
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Split into organizations (containing '/') and developers (no '/')
    orgs_df = df[df['i'].str.contains('/', na=False)].copy()
    devs_df = df[~df['i'].str.contains('/', na=False)].copy()

    # Apply transformations
    transformations = {
        'log': apply_log_transformation,
        'quantile': apply_quantile_transformation
    }

    base_name = Path(input_file).stem
    print(f"Processing {input_file}:")

    for transform_name, transform_func in transformations.items():
        # Apply transformation to both groups
        if len(orgs_df) > 0:
            orgs_transformed = transform_func(orgs_df)
        else:
            orgs_transformed = orgs_df.copy()

        if len(devs_df) > 0:
            devs_transformed = transform_func(devs_df)
        else:
            devs_transformed = devs_df.copy()

        # Generate output file names
        orgs_output = os.path.join(output_dir, f"{base_name}_orgs_{transform_name}.csv")
        devs_output = os.path.join(output_dir, f"{base_name}_devs_{transform_name}.csv")

        # Save the processed files
        orgs_transformed.to_csv(orgs_output, index=False)
        devs_transformed.to_csv(devs_output, index=False)

        # Show score ranges
        orgs_min = orgs_transformed['v'].min() if len(orgs_transformed) > 0 else 0
        orgs_max = orgs_transformed['v'].max() if len(orgs_transformed) > 0 else 0
        devs_min = devs_transformed['v'].min() if len(devs_transformed) > 0 else 0
        devs_max = devs_transformed['v'].max() if len(devs_transformed) > 0 else 0

        print(f"  - {transform_name.capitalize()} transformation:")
        print(f"    Organizations: {len(orgs_transformed)} entries -> {orgs_output}")
        print(f"    Score range: {orgs_min:.2f} - {orgs_max:.2f}")
        print(f"    Developers: {len(devs_transformed)} entries -> {devs_output}")
        print(f"    Score range: {devs_min:.2f} - {devs_max:.2f}")


def main():
    """
    Main function to process all score files
    """
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
    print()

    # Process each CSV file
    for csv_file in csv_files:
        try:
            split_and_process_scores(str(csv_file), output_dir)
            print()
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            print()

    print("Processing complete!")


if __name__ == "__main__":
    main()
