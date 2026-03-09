# Image Captioning

CNN encoder + LSTM decoder with Bahdanau attention for generating image captions.

## Architecture
- **Encoder**: CNN projection (simulates ResNet/EfficientNet features)
- **Attention**: Bahdanau soft-attention over 16 image regions
- **Decoder**: LSTM cell that attends to encoder features at each timestep

## Features
- Bahdanau attention mechanism
- Greedy caption generation
- Custom vocabulary (50 words)
- Teacher forcing during training
- Gradient clipping

## Setup

```bash
pip install -r requirements.txt
python main.py
```

Replace `CNNEncoder` with actual ResNet features for real image captioning.

## Output
- `training_loss.png` — cross-entropy loss curve
- `captioner.pth` — saved model weights
