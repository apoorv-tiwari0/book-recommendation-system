import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import re
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SEQ_LEN = 300
TOP_K = 10
CANDIDATES = 1000


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class BookRecommender:
    def __init__(self):
        # Load all artifacts ONCE
        self.tokenizer = pickle.load(open("artifacts/tokenizer/tokenizer.pkl", "rb"))
        self.model = tf.keras.models.load_model(
            "artifacts/models/lstm_encoder_triplet.keras",
            compile=False
        )
        self.embeddings = np.load("artifacts/embeddings/book_embeddings.npy")
        self.metadata = pd.read_csv("artifacts/embeddings/book_metadata.csv")

        self.vectorizer = pickle.load(open("artifacts/tfidf/vectorizer.pkl", "rb"))
        self.tfidf_matrix = pickle.load(open("artifacts/tfidf/tfidf_matrix.pkl", "rb"))

    def encode_query(self, query: str):
        seq = self.tokenizer.texts_to_sequences([query])
        seq = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding="post")
        vec = self.model.predict(seq, verbose=0)[0]
        return vec / (np.linalg.norm(vec) + 1e-8)

    def recommend(self, query: str, top_k: int = TOP_K):
        query = clean(query)

        # Stage 1: TF-IDF candidate selection
        q_tfidf = self.vectorizer.transform([query])
        scores = cosine_similarity(q_tfidf, self.tfidf_matrix)[0]
        top_idx = np.argsort(scores)[-CANDIDATES:]

        # Stage 2: Embedding re-ranking
        q_emb = self.encode_query(query)
        sims = self.embeddings[top_idx] @ q_emb
        best = top_idx[np.argsort(sims)[-top_k:][::-1]]

        results = self.metadata.iloc[best].copy()

        clean_results = []
        for _, row in results.iterrows():
            clean_results.append({
                "title": str(row.get("title", "")),
                "authors": str(row.get("authors", "")),
                "publication_year": (
                    int(row["publication_year"])
                    if pd.notna(row.get("publication_year"))
                    else None
                ),
                "language": (
                    str(row["language"])
                    if pd.notna(row.get("language"))
                    else None
                ),
            })

        return clean_results
