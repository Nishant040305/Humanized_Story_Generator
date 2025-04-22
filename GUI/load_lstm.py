import os
import torch
import torch.nn as nn
from transformers import GPT2TokenizerFast

# Model hyperparameters
SEQ_LEN = 100  # tokens per sequence window
BATCH_SIZE = 4
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.2
LR = 3e-4
EPOCHS = 5
CHECKPOINT_DIR = "../models"


class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        x = self.embed(x)
        x = self.dropout(x)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
            weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
        )


def load_model(checkpoint_path, tokenizer):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMGenerator(
        tokenizer.vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT
    ).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {checkpoint_path}")
    return model


def generate(model, tokenizer, prompt, max_len=100, temperature=1.0):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Convert prompt to token IDs
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    # Initialize hidden state
    hidden = model.init_hidden(1)
    hidden = tuple(h.to(DEVICE) for h in hidden)

    # Process the input sequence
    with torch.no_grad():
        for i in range(input_ids.size(1) - 1):
            _, hidden = model(input_ids[:, i].unsqueeze(1), hidden)

        # Generate new tokens
        curr_input = input_ids[:, -1].unsqueeze(1)
        output_ids = input_ids.clone()

        for _ in range(max_len):
            logits, hidden = model(curr_input, hidden)

            # Apply temperature
            logits = logits[:, -1, :] / temperature

            # Sample from the distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Add to output sequence
            output_ids = torch.cat([output_ids, next_token], dim=1)
            curr_input = next_token

            # Stop if we generate an EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main(prompt="Once upon a time"):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Create checkpoints directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Check for existing checkpoints
    checkpoint_files = [
        f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("model_epoch")
    ]

    if not checkpoint_files:
        raise FileNotFoundError(
            "No checkpoint files found. Please train the model first."
        )

    # Load latest checkpoint
    latest_checkpoint = max(
        checkpoint_files, key=lambda x: int(x.split("epoch")[1].split(".")[0])
    )
    checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
    model = load_model(checkpoint_path, tokenizer)

    temp = 1.0
    sample = generate(model, tokenizer, prompt, max_len=200, temperature=temp)
    # print(sample)
    return sample


if __name__ == "__main__":
    main()
