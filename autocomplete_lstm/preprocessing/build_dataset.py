import numpy as np
import pickle
from collections import Counter
from tqdm import tqdm

INPUT_FILE = "autocomplete_lstm/data/titles.txt"
OUTPUT_FILE = "autocomplete_lstm/data/dataset.npz"
TOKENIZER_PATH = "autocomplete_lstm/tokenizer/tokenizer.pkl"

MAX_VOCAB = 50000
SEQ_LEN = 5


def build_vocab(lines):
    counter = Counter()
    for line in lines:
        counter.update(line.split())

    most_common = counter.most_common(MAX_VOCAB - 2)

    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for i, (word, _) in enumerate(most_common, start=2):
        word2idx[word] = i

    return word2idx


def encode(line, word2idx):
    return [word2idx.get(w, 1) for w in line.split()]


def pad(seq):
    if len(seq) >= SEQ_LEN:
        return seq[-SEQ_LEN:]
    return [0] * (SEQ_LEN - len(seq)) + seq


def main():
    print("Loading titles...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = [l.strip().lower() for l in f if l.strip()]

    print(f"Total titles: {len(lines)}")

    print("Building vocab...")
    word2idx = build_vocab(lines)

    X, Y = [], []

    print("Generating samples...")

    for line in tqdm(lines):
        tokens = encode(line, word2idx)

        if len(tokens) < 3:
            continue

        for i in range(len(tokens) - 2):
            inp = tokens[:i+1]
            inp = pad(inp)

            target = tokens[i+1:i+3]

            if len(target) == 2:
                X.append(inp)
                Y.append(target)

    X = np.array(X, dtype=np.int32)
    Y = np.array(Y, dtype=np.int32)

    print("\nDataset created!")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    print("Saving dataset...")

    np.savez_compressed(OUTPUT_FILE, X=X, Y=Y)

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(word2idx, f)

    print("Saved successfully!")


if __name__ == "__main__":
    main()