import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os

# ===== CONFIG =====
DATA_PATH = "autocomplete_lstm/data/dataset.npz"
TOKENIZER_PATH = "autocomplete_lstm/tokenizer/tokenizer.pkl"
MODEL_PATH = "autocomplete_lstm/weights/lstm.pth"

BATCH_SIZE = 512   # safer for GPU memory
EMBED_SIZE = 128
HIDDEN_SIZE = 256
EPOCHS = 5
LR = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== DATASET =====
class TextDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ===== MODEL =====
class LSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_SIZE, HIDDEN_SIZE, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, vocab_size * 2)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)

        h = h[-1]  # last layer hidden state
        out = self.fc(h)

        out = out.view(-1, 2, self.vocab_size)
        return out


# ===== TRAIN =====
def main():
    print("Loading dataset...")
    data = np.load(DATA_PATH)
    X, Y = data["X"], data["Y"]

    print("Loading tokenizer...")
    with open(TOKENIZER_PATH, "rb") as f:
        word2idx = pickle.load(f)

    vocab_size = len(word2idx)
    print("Vocab size:", vocab_size)

    dataset = TextDataset(X, Y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMModel(vocab_size).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("Starting training on:", DEVICE)

    for epoch in range(EPOCHS):
        total_loss = 0

        for i, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            output = model(batch_x)

            loss1 = criterion(output[:, 0, :], batch_y[:, 0])
            loss2 = criterion(output[:, 1, :], batch_y[:, 1])
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 🔥 optional progress print every 100 batches
            if i % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {i} | Loss {loss.item():.4f}")

        print(f"\nEpoch {epoch+1}/{EPOCHS} Total Loss: {total_loss:.4f}\n")

    # create folder if not exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    main()