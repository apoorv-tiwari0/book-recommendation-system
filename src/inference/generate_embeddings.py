import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA_FILE = "data/processed/books_text_clean.csv"
TOKENIZER_PATH = "artifacts/tokenizer/tokenizer.pkl"
MODEL_PATH = "artifacts/models/lstm_encoder_triplet.keras"



EMBEDDING_DIR = "artifacts/embeddings"

MAX_SEQUENCE_LENGTH = 300
MAX_VOCAB_SIZE = 50000
BATCH_SIZE = 256


def main():
    os.makedirs(EMBEDDING_DIR, exist_ok=True)

    print("Loading tokenizer...")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    print("Loading trained encoder...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)

    texts = df["description"].astype(str).tolist()

    print("Tokenizing descriptions...")
    sequences = tokenizer.texts_to_sequences(texts)

    sequences = pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )

    print("Generating embeddings...")
    embeddings = []

    for i in range(0, len(sequences), BATCH_SIZE):
        batch = sequences[i:i + BATCH_SIZE]
        batch_embeddings = model.predict(batch, verbose=0)
        embeddings.append(batch_embeddings)

        if i % (BATCH_SIZE * 20) == 0:
            print(f"Processed {i} / {len(sequences)} books")

    embeddings = np.vstack(embeddings)

    print("L2-normalizing embeddings...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    print("Saving embeddings...")
    np.save(
        os.path.join(EMBEDDING_DIR, "book_embeddings.npy"),
        embeddings
    )

    print("Saving metadata...")
    metadata = df[
        ["book_id", "title", "authors", "publication_year", "language"]
    ]
    metadata.to_csv(
        os.path.join(EMBEDDING_DIR, "book_metadata.csv"),
        index=False
    )

    print("Embedding generation complete.")
    print(f"Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
