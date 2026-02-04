import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import re
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.sequence import pad_sequences

TOKENIZER_PATH = "artifacts/tokenizer/tokenizer.pkl"
MODEL_PATH = "artifacts/models/lstm_encoder.keras"
EMB_PATH = "artifacts/embeddings/book_embeddings.npy"
META_PATH = "artifacts/embeddings/book_metadata.csv"

TFIDF_VEC = "artifacts/tfidf/vectorizer.pkl"
TFIDF_MAT = "artifacts/tfidf/tfidf_matrix.pkl"

MAX_SEQ_LEN = 300
CANDIDATES = 1000
TOP_K = 10


def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_assets():
    tokenizer = pickle.load(open(TOKENIZER_PATH, "rb"))
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    embeddings = np.load(EMB_PATH)
    metadata = pd.read_csv(META_PATH)

    vectorizer = pickle.load(open(TFIDF_VEC, "rb"))
    tfidf_matrix = pickle.load(open(TFIDF_MAT, "rb"))

    return tokenizer, model, embeddings, metadata, vectorizer, tfidf_matrix


def encode_query(query, tokenizer, model):
    seq = tokenizer.texts_to_sequences([query])
    seq = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding="post")
    vec = model.predict(seq, verbose=0)[0]
    return vec / (np.linalg.norm(vec) + 1e-8)


def recommend(query):
    tokenizer, model, emb, meta, vec, tfidf = load_assets()

    query = clean(query)

    # Stage 1: TF-IDF retrieval
    q_tfidf = vec.transform([query])
    scores = cosine_similarity(q_tfidf, tfidf)[0]
    top_idx = np.argsort(scores)[-CANDIDATES:]

    # Stage 2: Embedding rerank
    q_emb = encode_query(query, tokenizer, model)
    sims = emb[top_idx] @ q_emb
    best = top_idx[np.argsort(sims)[-TOP_K:][::-1]]

    return meta.iloc[best]


if __name__ == "__main__":
    while True:
        q = input("\nQuery (exit to quit): ")
        if q == "exit":
            break

        results = recommend(q)
        for _, r in results.iterrows():
            print(f"{r.title} | {r.authors}")
