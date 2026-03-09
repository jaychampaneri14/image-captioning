"""
Image Captioning
CNN encoder (ResNet features) + LSTM decoder with attention for image captioning.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Simple vocabulary
VOCAB = ['<pad>', '<start>', '<end>', '<unk>',
         'a', 'an', 'the', 'is', 'are', 'in', 'on', 'with',
         'person', 'people', 'man', 'woman', 'child',
         'dog', 'cat', 'bird', 'car', 'bus', 'tree', 'flower',
         'red', 'blue', 'green', 'white', 'black', 'yellow',
         'sitting', 'standing', 'running', 'playing', 'eating',
         'large', 'small', 'two', 'group', 'outdoor', 'indoor',
         'street', 'park', 'grass', 'sky', 'water', 'table']

W2I = {w: i for i, w in enumerate(VOCAB)}
I2W = {i: w for w, i in W2I.items()}
VOCAB_SIZE = len(VOCAB)

# Sample captions paired with "image features" (we simulate with random vectors)
CAPTION_TEMPLATES = [
    "a person is standing in the park",
    "two dogs are playing on the grass",
    "a red car is on the street",
    "a woman is sitting with a cat",
    "a group of people are outdoor",
    "a large bird is sitting on a tree",
    "a man is eating with a dog",
    "a child is running in the park",
    "two people are standing on the street",
    "a small cat is playing indoor",
]


class CNNEncoder(nn.Module):
    """Simulated CNN encoder — in real use, replace with ResNet/EfficientNet features."""
    def __init__(self, embed_dim=256):
        super().__init__()
        # Simulate CNN output: feature map of (B, C, H, W)
        self.projection = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, visual_features):
        # visual_features: (B, 512) pretrained CNN feature
        return self.projection(visual_features)


class Attention(nn.Module):
    """Bahdanau attention over image features."""
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att    = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (B, num_regions, encoder_dim)
        # decoder_hidden: (B, decoder_dim)
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        att  = torch.tanh(att1 + att2)
        alpha = torch.softmax(self.full_att(att).squeeze(2), dim=1)
        context = (encoder_out * alpha.unsqueeze(2)).sum(1)
        return context, alpha


class LSTMDecoder(nn.Module):
    """LSTM decoder with attention for caption generation."""
    def __init__(self, vocab_size, embed_dim=128, decoder_dim=512,
                 encoder_dim=256, attention_dim=256):
        super().__init__()
        self.embed      = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention  = Attention(encoder_dim, decoder_dim, attention_dim)
        self.lstm_cell  = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.init_h     = nn.Linear(encoder_dim, decoder_dim)
        self.init_c     = nn.Linear(encoder_dim, decoder_dim)
        self.fc_out     = nn.Linear(decoder_dim, vocab_size)
        self.dropout    = nn.Dropout(0.3)
        self.decoder_dim = decoder_dim

    def init_hidden_state(self, encoder_out):
        mean_enc = encoder_out.mean(dim=1)
        h = torch.tanh(self.init_h(mean_enc))
        c = torch.tanh(self.init_c(mean_enc))
        return h, c

    def forward(self, encoder_out, captions, lengths):
        """
        encoder_out: (B, num_regions, encoder_dim)
        captions:    (B, max_len)  — token indices
        lengths:     (B,) — caption lengths
        """
        B = captions.size(0)
        embeddings = self.dropout(self.embed(captions))  # (B, max_len, embed_dim)
        h, c = self.init_hidden_state(encoder_out)

        max_len = max(lengths)
        predictions = torch.zeros(B, max_len, VOCAB_SIZE).to(encoder_out.device)
        alphas      = torch.zeros(B, max_len, encoder_out.size(1)).to(encoder_out.device)

        for t in range(max_len):
            context, alpha = self.attention(encoder_out, h)
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            pred = self.fc_out(self.dropout(h))
            predictions[:, t, :] = pred
            alphas[:, t, :] = alpha

        return predictions, alphas


class ImageCaptioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder(embed_dim=256)
        self.decoder = LSTMDecoder(VOCAB_SIZE)

    def forward(self, visual_feats, captions, lengths):
        # Simulate spatial features: repeat for 16 "regions"
        enc = self.encoder(visual_feats).unsqueeze(1).expand(-1, 16, -1)
        return self.decoder(enc, captions, lengths)

    def generate(self, visual_feats, max_len=20, device='cpu'):
        """Greedy caption generation."""
        self.eval()
        with torch.no_grad():
            enc = self.encoder(visual_feats).unsqueeze(1).expand(-1, 16, -1)
            B   = visual_feats.size(0)
            h, c = self.decoder.init_hidden_state(enc)
            word = torch.full((B,), W2I['<start>'], dtype=torch.long).to(device)
            generated = []
            for _ in range(max_len):
                emb  = self.decoder.embed(word)
                ctx, _ = self.decoder.attention(enc, h)
                h, c = self.decoder.lstm_cell(torch.cat([emb, ctx], dim=1), (h, c))
                logits = self.decoder.fc_out(h)
                word   = logits.argmax(1)
                generated.append(word.item())
                if word.item() == W2I['<end>']:
                    break
        return [I2W.get(w, '<unk>') for w in generated if w not in [W2I['<pad>'], W2I['<end>']]]


def tokenize_caption(caption):
    tokens = ['<start>'] + caption.split() + ['<end>']
    return [W2I.get(t, W2I['<unk>']) for t in tokens]


def generate_dataset(n_samples=500, seed=42):
    torch.manual_seed(seed)
    captions = [CAPTION_TEMPLATES[i % len(CAPTION_TEMPLATES)] for i in range(n_samples)]
    tokens   = [tokenize_caption(c) for c in captions]
    max_len  = max(len(t) for t in tokens)
    padded   = torch.zeros(n_samples, max_len, dtype=torch.long)
    lengths  = []
    for i, t in enumerate(tokens):
        padded[i, :len(t)] = torch.LongTensor(t)
        lengths.append(len(t))
    visual_feats = torch.randn(n_samples, 512)
    return visual_feats, padded, torch.LongTensor(lengths)


def train(model, loader, optimizer, criterion, device, epochs=30):
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for vf, caps, lens in loader:
            vf, caps, lens = vf.to(device), caps.to(device), lens.to(device)
            optimizer.zero_grad()
            preds, _ = model(vf, caps[:, :-1], (lens - 1).tolist())
            # Compute loss over all time steps
            tgt = caps[:, 1:preds.size(1)+1]
            loss = criterion(preds.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: Loss={losses[-1]:.4f}")
    return losses


def main():
    print("=" * 60)
    print("IMAGE CAPTIONING — CNN+LSTM with Attention")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, Vocab: {VOCAB_SIZE} words")

    vf, caps, lens = generate_dataset(500)
    ds = TensorDataset(vf, caps, lens)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model     = ImageCaptioner().to(device)
    params    = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding

    print("\n--- Training ---")
    losses = train(model, loader, optimizer, criterion, device, epochs=40)

    # Generate captions for test images
    print("\n--- Generated Captions ---")
    model.eval()
    test_vf = torch.randn(5, 512).to(device)
    for i in range(5):
        caption_words = model.generate(test_vf[i:i+1], device=device)
        caption = ' '.join(caption_words)
        print(f"  Image {i+1}: {caption}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(losses, 'b-'); plt.title('Captioning Model Training Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('training_loss.png', dpi=150); plt.close()

    torch.save(model.state_dict(), 'captioner.pth')
    print("\nModel saved to captioner.pth")
    print("\n✓ Image Captioning complete!")


if __name__ == '__main__':
    main()
