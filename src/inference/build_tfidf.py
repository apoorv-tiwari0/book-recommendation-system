import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_FILE = "data/processed/books_text_clean.csv"
OUT_DIR = "artifacts/tfidf"

MAX_FEATURES = 50000


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    texts = df["description"].astype(str).tolist()

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    with open(f"{OUT_DIR}/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open(f"{OUT_DIR}/tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)

    print("TF-IDF index built.")
    print("Shape:", tfidf_matrix.shape)


if __name__ == "__main__":
    main()
