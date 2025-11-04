#!/usr/bin/env python3
"""
Generate JSON files for UI from seed and output data.

This script:
1. Loads all entries from seed/ directory
2. Loads corresponding developer score files from output/ directory (devs only)
3. Creates JSON files for each ecosystem with format:
   {
     "category": "devrank",
     "ecosystem": "<ecosystem_name>",
     "seed": [{"i": "peer_id", "v": value}, ...],
     "scores": [{"i": "peer_id", "v": value}, ...]
   }
4. Saves JSON files to ui/ directory

Usage:
    python3 generate_json.py

Requirements:
    - pandas (install with: pip install pandas)
    - CSV files in seed/ directory
    - CSV files in output/ directory (matching seed filenames with _devs suffix)

Output:
    - Creates ui/ directory if it doesn't exist
    - For each seed file (e.g., bitcoin.csv), creates:
      - ui/bitcoin.json with seed and score data
"""

import os
import json
import pandas as pd
from pathlib import Path


def capitalize_ecosystem_name(filename):
    """
    Convert filename to capitalized ecosystem name.

    Examples:
        bitcoin -> Bitcoin
        protocol-guild -> Protocol-Guild
        cryptoxai -> Cryptoxai

    Args:
        filename (str): Base filename without extension

    Returns:
        str: Capitalized ecosystem name
    """
    # Split on hyphens and capitalize each part
    parts = filename.split("-")
    capitalized = "-".join(word.capitalize() for word in parts)
    return capitalized


def load_seed_data(seed_file):
    """
    Load all entries from seed file.

    Args:
        seed_file (Path): Path to seed CSV file

    Returns:
        list: List of dictionaries with 'i' and 'v' keys
    """
    df = pd.read_csv(seed_file)

    # Convert to list of dictionaries
    seed_data = df.to_dict("records")

    return seed_data


def load_scores(scores_file):
    """
    Load scores from output file.

    Args:
        scores_file (Path): Path to scores CSV file

    Returns:
        list: List of dictionaries with 'i' and 'v' keys
    """
    if not scores_file.exists():
        print(f"Warning: {scores_file} not found, using empty scores")
        return []

    df = pd.read_csv(scores_file)

    # Convert to list of dictionaries
    scores_data = df.to_dict("records")

    return scores_data


def generate_json_file(ecosystem_name, seed_data, scores_data, output_file):
    """
    Generate JSON file with seed and scores data.

    Args:
        ecosystem_name (str): Name of the ecosystem
        seed_data (list): List of seed entries
        scores_data (list): List of score entries
        output_file (Path): Path to output JSON file
    """
    json_data = {
        "category": "devrank",
        "ecosystem": ecosystem_name,
        "seed": seed_data,
        "scores": scores_data,
    }

    # Write JSON file with pretty formatting
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Created {output_file}")
    print(f"  Seed entries: {len(seed_data)}")
    print(f"  Score entries: {len(scores_data)}")


def main():
    """
    Main execution function.
    """
    print("=" * 60)
    print("Generating JSON files for UI")
    print("=" * 60)
    print()

    # Define directories
    seed_dir = Path("seed")
    output_dir = Path("output")
    ui_dir = Path("ui")

    # Create ui directory if it doesn't exist
    ui_dir.mkdir(exist_ok=True)
    print(f"✓ Output directory: {ui_dir}/")
    print()

    # Find all seed CSV files
    seed_files = sorted(list(seed_dir.glob("*.csv")))

    if not seed_files:
        print(f"❌ No CSV files found in {seed_dir}")
        return

    print(f"Found {len(seed_files)} seed file(s) to process...")
    print()

    # Process each seed file
    for seed_file in seed_files:
        base_name = seed_file.stem
        ecosystem_name = capitalize_ecosystem_name(base_name)

        print(f"Processing: {base_name}")

        # Load seed data (all entries)
        seed_data = load_seed_data(seed_file)

        # Find corresponding scores file
        scores_file = output_dir / f"{base_name}_devs.csv"
        scores_data = load_scores(scores_file)

        # Generate JSON file
        output_file = ui_dir / f"{base_name}.json"
        generate_json_file(ecosystem_name, seed_data, scores_data, output_file)
        print()

    print("=" * 60)
    print("✓ JSON generation complete!")
    print("=" * 60)
    print(f"\nJSON files saved to {ui_dir}/")


if __name__ == "__main__":
    main()
