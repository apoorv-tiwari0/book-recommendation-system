import re
import os
import pandas as pd

INPUT_FILE = "data/processed/books_clean.csv"
OUTPUT_FILE = "data/processed/books_text_clean.csv"

MIN_WORDS = 30
MAX_WORDS = 300


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Keep alphabets and basic punctuation
    text = re.sub(r"[^a-zA-Z\s.,!?']", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def filter_and_truncate(text: str) -> str | None:
    words = text.split()

    if len(words) < MIN_WORDS:
        return None

    if len(words) > MAX_WORDS:
        words = words[:MAX_WORDS]

    return " ".join(words)


def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv(INPUT_FILE)

    print("Cleaning text...")
    df["description"] = df["description"].astype(str).apply(clean_text)

    print("Filtering and truncating descriptions...")
    df["description"] = df["description"].apply(filter_and_truncate)

    initial_size = len(df)
    df = df.dropna(subset=["description"])
    final_size = len(df)

    print(f"Rows before filtering: {initial_size}")
    print(f"Rows after filtering: {final_size}")

    df = df.reset_index(drop=True)

    print("Saving cleaned text dataset...")
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved to {OUTPUT_FILE}")

    pd.read_csv("data/processed/books_text_clean.csv").sample(3)["description"]



if __name__ == "__main__":
    main()
