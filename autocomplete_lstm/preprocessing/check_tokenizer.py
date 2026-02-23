from collections import Counter

INPUT_FILE = "autocomplete_lstm/data/titles.txt"
MAX_VOCAB = 50000


def main():
    print("Loading titles...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = [l.strip().lower() for l in f if l.strip()]

    print(f"Total titles: {len(lines)}")

    # build vocab
    counter = Counter()
    for line in lines:
        counter.update(line.split())

    most_common = counter.most_common(MAX_VOCAB - 2)

    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for i, (word, _) in enumerate(most_common, start=2):
        word2idx[word] = i

    print(f"Vocab size: {len(word2idx)}")

    # show sample words
    print("\nTop 10 words:")
    for w, c in most_common[:10]:
        print(w, "→", c)

    # test encoding
    sample = lines[0]
    print("\nSample title:", sample)

    encoded = [word2idx.get(w, 1) for w in sample.split()]
    print("Encoded:", encoded[:10])

    # decode back
    idx2word = {v: k for k, v in word2idx.items()}
    decoded = [idx2word.get(i, "<UNK>") for i in encoded]

    print("Decoded:", decoded[:10])


if __name__ == "__main__":
    main()