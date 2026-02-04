import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lstm_encoder import build_lstm_encoder

DATA_FILE = "data/processed/books_text_clean.csv"
TOKENIZER_PATH = "artifacts/tokenizer/tokenizer.pkl"
MODEL_DIR = "artifacts/models"

MAX_SEQUENCE_LENGTH = 300
CHUNK_LEN = 200
BATCH_SIZE = 64
EPOCHS = 3
MAX_VOCAB_SIZE = 50000
MARGIN = 0.3


def triplet_loss(anchor, positive, negative):
    anchor = tf.math.l2_normalize(anchor, axis=1)
    positive = tf.math.l2_normalize(positive, axis=1)
    negative = tf.math.l2_normalize(negative, axis=1)

    pos_sim = tf.reduce_sum(anchor * positive, axis=1)
    neg_sim = tf.reduce_sum(anchor * negative, axis=1)

    loss = tf.maximum(0.0, MARGIN - pos_sim + neg_sim)
    return tf.reduce_mean(loss)


def batch_generator(sequences, batch_size):
    n = len(sequences)

    while True:
        anchor_idx = np.random.choice(n, batch_size)
        neg_idx = np.random.choice(n, batch_size)

        anchor_batch = sequences[anchor_idx]
        negative_batch = sequences[neg_idx]

        a = np.zeros((batch_size, MAX_SEQUENCE_LENGTH), dtype=np.int32)
        p = np.zeros((batch_size, MAX_SEQUENCE_LENGTH), dtype=np.int32)
        n_batch = np.zeros((batch_size, MAX_SEQUENCE_LENGTH), dtype=np.int32)

        a[:, :CHUNK_LEN] = anchor_batch[:, :CHUNK_LEN]
        p[:, :CHUNK_LEN] = anchor_batch[:, -CHUNK_LEN:]
        n_batch[:, :CHUNK_LEN] = negative_batch[:, :CHUNK_LEN]

        yield a, p, n_batch


def main():
    print("Loading tokenizer...")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    texts = df["description"].astype(str).tolist()

    print("Tokenizing...")
    sequences = tokenizer.texts_to_sequences(texts)
    sequences = pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )

    vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)

    model = build_lstm_encoder(vocab_size=vocab_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    print("Starting triplet training...")

    gen = batch_generator(sequences, BATCH_SIZE)

    for epoch in range(EPOCHS):
        for step in range(600):
            a, p, n_batch = next(gen)

            with tf.GradientTape() as tape:
                ea = model(a, training=True)
                ep = model(p, training=True)
                en = model(n_batch, training=True)

                loss = triplet_loss(ea, ep, en)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if step % 50 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.numpy():.4f}")

    model.save(f"{MODEL_DIR}/lstm_encoder_triplet.keras")
    print("Triplet-trained model saved.")


if __name__ == "__main__":
    main()
