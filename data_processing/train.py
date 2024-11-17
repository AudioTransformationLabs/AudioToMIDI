import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram, MFCC
import torch.nn as nn
import torch.optim as optim
from .dataset import AudioMidiDataset
from .model import AudioToMidiCNN
from .constants import *
from .transformer import Transformer

transform = Transformer.mel_spec_transform() if FEATURE_TYPE == "mel_spec" else Transformer.mfcc_transform()

train_dataset = AudioMidiDataset(TRAIN_AUDIO_PATH, TRAIN_MIDI_PATH, transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = AudioToMidiCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    model.train()
    curr_loss = 0.0
    for batch_idx, (features, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx+1}/{len(train_loader)}")
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        curr_loss += loss.item()

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {curr_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print("Model saved successfully.")
