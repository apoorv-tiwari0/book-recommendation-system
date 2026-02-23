import pandas as pd
import re
from pathlib import Path

# Paths
INPUT_CSV = "data/processed/books_text_clean.csv"
OUTPUT_TXT = "autocomplete_lstm/data/titles.txt"

# Ensure output directory exists
Path(OUTPUT_TXT).parent.mkdir(parents=True, exist_ok=True)

def clean_title(title: str) -> str:
    title = str(title).lower()
    title = re.sub(r"[^a-z0-9\s']", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title

def main():
    print("Loading CSV...")
    df = pd.read_csv(INPUT_CSV, usecols=["title"])

    print("Cleaning titles...")
    titles = (
        df["title"]
        .dropna()
        .astype(str)
        .map(clean_title)
        .drop_duplicates()
    )

    print(f"Saving {len(titles)} titles...")
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for t in titles:
            if len(t.split()) >= 2:
                f.write(t + "\n")

    print("Done.")

if __name__ == "__main__":
    main()
