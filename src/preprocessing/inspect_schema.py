import os
import pandas as pd

RAW_DATA_DIR = "data/raw"

def inspect_csv_headers():
    print("Inspecting CSV schemas...\n")

    for file in sorted(os.listdir(RAW_DATA_DIR)):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(RAW_DATA_DIR, file)

        try:
            df = pd.read_csv(file_path, nrows=5)
            print(f"File: {file}")
            print(f"   Columns ({len(df.columns)}):")
            for col in df.columns:
                print(f"   - {col}")
            print("-" * 50)

        except Exception as e:
            print(f"Error reading {file}: {e}")
            print("-" * 50)


if __name__ == "__main__":
    inspect_csv_headers()
