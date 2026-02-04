import os
import pandas as pd

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
OUTPUT_FILE = "books_clean.csv"

# Canonical column mapping
COLUMN_MAPPING = {
    "Id": "book_id",
    "Name": "title",
    "Authors": "authors",
    "Description": "description",
    "PublishYear": "publication_year",
    "Language": "language"
}

CANONICAL_COLUMNS = list(COLUMN_MAPPING.values())


def load_and_normalize():
    all_dfs = []

    print(" Loading and normalizing CSV files...\n")

    for file in sorted(os.listdir(RAW_DATA_DIR)):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(RAW_DATA_DIR, file)
        print(f"➡ Processing: {file}")

        df = pd.read_csv(file_path)

        # Keep only columns we care about
        available_cols = [col for col in COLUMN_MAPPING.keys() if col in df.columns]
        df = df[available_cols]

        # Rename columns to canonical names
        df = df.rename(columns=COLUMN_MAPPING)

        all_dfs.append(df)

    print("\n Concatenating all dataframes...")
    merged_df = pd.concat(all_dfs, ignore_index=True)

    return merged_df


def clean_dataframe(df):
    print("🧹 Cleaning data...")

    # Drop rows with missing or empty descriptions
    df = df.dropna(subset=["description"])
    df = df[df["description"].str.strip() != ""]

    # Optional: drop rows with missing title
    df = df.dropna(subset=["title"])

    # Reset index
    df = df.reset_index(drop=True)

    print(f"Final dataset size: {len(df)} rows")
    return df


def save_dataframe(df):
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, OUTPUT_FILE)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to: {output_path}")


if __name__ == "__main__":
    df = load_and_normalize()
    df = clean_dataframe(df)
    save_dataframe(df)
