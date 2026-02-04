import os
import json
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

INPUT_FILE = "data/processed/books_text_clean.csv"
TOKENIZER_DIR = "artifacts/tokenizer"

MAX_VOCAB_SIZE = 50000
MAX_SEQUENCE_LENGTH = 300
OOV_TOKEN = "<OOV>"


def build_tokenizer(texts):
    tokenizer = Tokenizer(
        num_words=MAX_VOCAB_SIZE,
        oov_token=OOV_TOKEN
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer


def save_tokenizer(tokenizer):
    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    with open(os.path.join(TOKENIZER_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    config = {
        "max_vocab_size": MAX_VOCAB_SIZE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "oov_token": OOV_TOKEN,
        "actual_vocab_size": len(tokenizer.word_index)
    }

    with open(os.path.join(TOKENIZER_DIR, "tokenizer_config.json"), "w") as f:
        json.dump(config, f, indent=4)


def main():
    print("Loading cleaned text dataset...")
    df = pd.read_csv(INPUT_FILE)

    texts = df["description"].astype(str).tolist()

    print("Building tokenizer...")
    tokenizer = build_tokenizer(texts)

    print("Converting texts to sequences...")
    sequences = tokenizer.texts_to_sequences(texts)

    print("Padding sequences...")
    padded_sequences = pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )

    print("Saving tokenizer artifacts...")
    save_tokenizer(tokenizer)

    print("Tokenizer build complete")
    print(f"Vocabulary size (actual): {len(tokenizer.word_index)}")
    print(f"Padded sequence shape: {padded_sequences.shape}")


if __name__ == "__main__":
    main()
