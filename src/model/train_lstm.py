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
BATCH_SIZE = 64
EPOCHS = 3
MAX_VOCAB_SIZE = 50000


def cosine_loss(a, b):
    a = tf.math.l2_normalize(a, axis=1, epsilon=1e-8)
    b = tf.math.l2_normalize(b, axis=1, epsilon=1e-8)
    return 1 - tf.reduce_mean(tf.reduce_sum(a * b, axis=1))



def batch_generator(sequences, batch_size):
    while True:
        idx = np.random.choice(len(sequences), batch_size)
        batch = sequences[idx]

        x1 = np.zeros((batch_size, MAX_SEQUENCE_LENGTH), dtype=np.int32)
        x2 = np.zeros((batch_size, MAX_SEQUENCE_LENGTH), dtype=np.int32)

        x1[:, :200] = batch[:, :200]
        x2[:, :200] = batch[:, -200:]

        yield x1, x2



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

    print("Starting training...")

    for epoch in range(EPOCHS):
        gen = batch_generator(sequences, BATCH_SIZE)

        for step in range(500):  # fixed steps per epoch
            x1, x2 = next(gen)

            with tf.GradientTape() as tape:
                e1 = model(x1, training=True)
                e2 = model(x2, training=True)
                loss = cosine_loss(e1, e2)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if step % 50 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.numpy():.4f}")

    model.save(f"{MODEL_DIR}/lstm_encoder.keras")
    print("Model saved.")


if __name__ == "__main__":
    main()
