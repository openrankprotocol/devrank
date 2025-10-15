#!/usr/bin/env python3
"""
Score Processing Script

This script processes score files from the scores/ directory by:
1. Loading all CSV score files
2. Splitting peers into organizations (containing '/') and developers (no '/')
3. Normalizing scores within each group so all scores sum to 1
4. Saving results to output/ directory with _orgs and _devs suffixes

Usage:
    python3 process_scores.py

Requirements:
    - pandas (install with: pip install pandas)
    - CSV files in scores/ directory with columns 'i' (identifier) and 'v' (score)

Output:
    - Creates output/ directory if it doesn't exist
    - For each input file (e.g., ai.csv), creates:
      - {filename}_orgs.csv: Organizations (identifiers containing '/')
      - {filename}_devs.csv: Developers (identifiers without '/')
    - Scores are normalized within each group (sum to 1) and sorted by score (descending)
"""

import os
import pandas as pd
from pathlib import Path


def normalize_scores(df):
    """
    Normalize scores so that all scores sum to 1

    Args:
        df (pandas.DataFrame): DataFrame with 'v' column containing scores

    Returns:
        pandas.DataFrame: DataFrame with normalized scores
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

    return df_normalized


def split_and_process_scores(input_file, output_dir):
    """
    Process a single score file by splitting into orgs and devs, normalizing, and saving

    Args:
        input_file (str): Path to input CSV file
        output_dir (str): Directory to save processed files
    """
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Split into organizations (containing '/') and developers (no '/')
    orgs_df = df[df['i'].str.contains('/', na=False)].copy()
    devs_df = df[~df['i'].str.contains('/', na=False)].copy()

    # Normalize scores within each group
    orgs_normalized = normalize_scores(orgs_df)
    devs_normalized = normalize_scores(devs_df)

    # Sort by score (descending)
    orgs_normalized = orgs_normalized.sort_values('v', ascending=False)
    devs_normalized = devs_normalized.sort_values('v', ascending=False)

    # Generate output file names
    base_name = Path(input_file).stem
    orgs_output = os.path.join(output_dir, f"{base_name}_orgs.csv")
    devs_output = os.path.join(output_dir, f"{base_name}_devs.csv")

    # Save the processed files
    orgs_normalized.to_csv(orgs_output, index=False)
    devs_normalized.to_csv(devs_output, index=False)

    # Verify normalization
    orgs_sum = orgs_normalized['v'].sum()
    devs_sum = devs_normalized['v'].sum()

    print(f"Processed {input_file}:")
    print(f"  - Organizations: {len(orgs_normalized)} entries -> {orgs_output}")
    print(f"    Sum of scores: {orgs_sum:.10f}")
    print(f"  - Developers: {len(devs_normalized)} entries -> {devs_output}")
    print(f"    Sum of scores: {devs_sum:.10f}")


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
