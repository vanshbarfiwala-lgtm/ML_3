import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import pickle

# --- Load vocabularies ---
code_vocab_data = torch.load("vocab_dicts.pth", map_location="cpu")
code_token_to_id = code_vocab_data["token_to_id"]
code_id_to_token = code_vocab_data["id_to_token"]
code_vocab_size = len(code_token_to_id)

with open("vocab_nl.pkl", "rb") as f:
    nl_vocab_data = pickle.load(f)
nl_token_to_id = nl_vocab_data[0]
nl_id_to_token = {v: k for k, v in nl_token_to_id.items()}
nl_vocab_size = len(nl_token_to_id)


# --- Model Definitions ---
class MLPModel(nn.Module):
    def __init__(self, vocab_size, context_window=12, embedding_dim=64, hidden_size=512, activation="relu", dropout_p=0.37):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_window * embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_p)
        self.activation = nn.ReLU() if activation.lower() == "relu" else nn.Tanh()

    def forward(self, x):
        embeds = self.embedding(x)
        embeds = embeds.view(embeds.size(0), -1)
        h = self.dropout(self.activation(self.fc1(embeds)))
        h = self.dropout(self.activation(self.fc2(h)))
        logits = self.out(h)
        return logits


class MLPTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=1024, seq_len=12, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(embed_dim * seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim // 2, vocab_size)
        self.relu = nn.ReLU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        embeds = self.embed_dropout(self.embedding(x))
        flat = embeds.view(embeds.size(0), -1)
        h = self.relu(self.fc1(flat))
        h = self.dropout(h)
        h = self.relu(self.fc2(h))
        h = self.dropout(h)
        out = self.fc_out(h)
        return out


# --- Tokenization ---
def tokenize_code_line(line):
    line = re.sub(r'//.*', '', line)
    line = re.sub(r'/\*.*?\*/', '', line)
    tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*|[{}()\[\];,.=<>!+\-*/&|^%#~?:]', line)
    return tokens


def tokenize_nl_line(line):
    return line.strip().split()


# --- Sequence Generation ---
def generate_sequence(model, start_text, token_to_id, id_to_token, max_new_tokens=20,
                      context_window=12, device="cpu", category="code", approach="greedy", temperature=1.0):
    model.eval()
    tokens = tokenize_code_line(start_text) if category == "code" else tokenize_nl_line(start_text)
    context = [token_to_id.get(t, 0) for t in tokens]
    generated = tokens.copy()

    for _ in range(max_new_tokens):
        x = context[-context_window:]
        if len(x) < context_window:
            x = [0] * (context_window - len(x)) + x
        x_tensor = torch.tensor([x], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(x_tensor)[0] / temperature
            probs = F.softmax(logits, dim=-1)

            if approach == "greedy":
                next_id = torch.argmax(probs).item()
            else:
                next_id = torch.multinomial(probs, 1).item()

        next_token = id_to_token.get(next_id, "<unk>")
        generated.append(next_token)
        if next_token == "<eos>":
            generated.pop()
            break
        
        context.append(next_id)

    return " ".join(generated)


# --- Streamlit UI ---
st.set_page_config(page_title="Next-Token Visualizer", layout="wide")

with st.sidebar:
    st.title("Model Configuration")

    category = st.selectbox(
        "Select Category",
        ["Code Prediction", "Natural Language"],
        help="Choose whether to generate programming code or natural language text."
    )

    approach = st.selectbox(
        "Generation Approach",
        ["Greedy", "Sampling"],
        help="Greedy picks the most likely next token; Sampling adds randomness for diversity."
    )

    context_size = st.selectbox(
        "Context Size",
        [6, 12],
        help="The number of previous tokens the model looks at to predict the next one."
    )

    embedding_dim = st.selectbox(
        "Embedding Dimension",
        [32, 64],
        help="The size of the vector representing each token."
    )

    activation = st.selectbox(
        "Activation Function",
        ["ReLU", "Tanh"],
        help="Determines how signals are transformed inside the network layers."
    )

    temperature = st.slider(
        "Temperature",
        0.5, 2.0, 1.0, 0.1,
        help="Controls randomness in generation. Lower = more deterministic; higher = more creative."
    )

    st.markdown("---")


# --- Main Screen ---
st.title("Next-Token Visualizer")

if category == "Code Prediction":
    st.markdown("""
    **Code Prediction Mode:**  
    The model will predict the next few lines of code on the input.  
    You can try with something like " if ( == ".
    """)
else:
    st.markdown("""
    **Natural Language Mode:**  
    The model will continue your input text. 
    You can try starting with short phrase.
    For example *"World is "* 
    """)

prompt = st.text_area("Enter your starting sequence:", "")

if st.button("Generate Prediction"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if category == "Code Prediction":
        vocab_size = code_vocab_size
        token_to_id = code_token_to_id
        id_to_token = code_id_to_token
        model_filename = f"model_{context_size}_emb{embedding_dim}_{activation.lower()}.pth"
        model = MLPModel(vocab_size, context_window=context_size, embedding_dim=embedding_dim, activation=activation).to(device)
    else:
        vocab_size = nl_vocab_size
        token_to_id = nl_token_to_id
        id_to_token = nl_id_to_token
        model_filename = f"mod_{context_size}_{embedding_dim}_{activation.lower()}.pth"
        model = MLPTextGenerator(vocab_size=vocab_size, embed_dim=embedding_dim, hidden_dim=1024, seq_len=context_size).to(device)

    try:
        model.load_state_dict(torch.load(model_filename, map_location=device))
        st.success(f"Loaded model: {model_filename}")
    except Exception as e:
        st.warning(f"Could not load model: {e}")

    generated_text = generate_sequence(
        model, prompt, token_to_id, id_to_token,
        max_new_tokens=40, context_window=context_size,
        device=device, category="code" if category == "Code Prediction" else "nl",
        approach=approach.lower(), temperature=temperature
    )

    st.subheader("Predicted Continuation:")
    st.code(generated_text, language="c" if category == "Code Prediction" else "text")
