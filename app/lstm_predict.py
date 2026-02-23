import torch
import torch.nn as nn
import pickle

MODEL_PATH = "autocomplete_lstm/weights/lstm.pth"
TOKENIZER_PATH = "autocomplete_lstm/tokenizer/tokenizer.pkl"

SEQ_LEN = 5
EMBED_SIZE = 128
HIDDEN_SIZE = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        h = h[-1]
        out = self.fc(h)

        out = out.view(-1, 2, self.vocab_size)
        return out


# LOAD ONCE
with open(TOKENIZER_PATH, "rb") as f:
    word2idx = pickle.load(f)

idx2word = {v: k for k, v in word2idx.items()}
vocab_size = len(word2idx)

model = LSTMModel(vocab_size).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


def pad(seq):
    if len(seq) >= SEQ_LEN:
        return seq[-SEQ_LEN:]
    return [0] * (SEQ_LEN - len(seq)) + seq


def predict_next_words(text: str):
    text = text.lower().strip()

    tokens = [word2idx.get(w, 1) for w in text.split()]
    tokens = pad(tokens)

    x = torch.tensor([tokens], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        output = model(x)

    pred1 = torch.argmax(output[0, 0]).item()
    w1 = idx2word.get(pred1, "")

    # SECOND WORD (conditioned)
    new_tokens = tokens + [pred1]
    new_tokens = new_tokens[-SEQ_LEN:]

    x2 = torch.tensor([new_tokens], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        output2 = model(x2)

    pred2 = torch.argmax(output2[0, 0]).item()
    w2 = idx2word.get(pred2, "")

    words = [w for w in [w1, w2] if w and w != "<UNK>"]

    return " ".join(words)