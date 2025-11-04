import torch
import torch.nn as nn

class MLPTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=1024, seq_len=5, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(0.1)

        # --- Two hidden layers ---
        self.fc1 = nn.Linear(embed_dim * seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim // 2, vocab_size)
        self.relu = nn.ReLU()

        # --- Weight initialization ---
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        embeds = self.embed_dropout(self.embedding(x))

        # Flatten embeddings for MLP input
        flat = embeds.view(embeds.size(0), -1)

        # --- First hidden layer ---
        h = self.relu(self.fc1(flat))
        h = self.dropout(h)

        # --- Second hidden layer ---
        h = self.relu(self.fc2(h))
        h = self.dropout(h)

        # --- Output ---
        out = self.fc_out(h)
        return out
    
import torch
import pickle

# --- Load vocab mappings ---
with open("vocab_nl.pkl", "rb") as f:
    word_to_idx, idx_to_word = pickle.load(f)
print("✅ Vocabulary loaded!")

# --- Recreate model (same parameters as training) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPTextGenerator(
    vocab_size=len(word_to_idx),
    embed_dim=32,
    hidden_dim=1024,
    seq_len=12
).to(device)

# --- Load model weights ---
model.load_state_dict(torch.load("mod_12_32_tanh.pth", map_location=device))
model.eval()
print("✅ Model loaded and ready for inference!")