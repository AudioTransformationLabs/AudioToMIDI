import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .constants import (
    BATCH_SIZE,
    FEATURE_TYPE,
    MODEL_PATH,
    NUM_EPOCHS,
    TRAIN_AUDIO_PATH,
    TRAIN_MIDI_PATH,
    TEST_AUDIO_PATH,
    TEST_MIDI_PATH,
)
from .dataset import AudioMidiDataset
from .evaluate import evaluate_model
from .model import AudioToMidiCNN
from .transformer import Transformer

transform = (
    Transformer.mel_spec_transform()
    if FEATURE_TYPE == "mel_spec"
    else Transformer.mfcc_transform()
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} to run model training")
train_dataset = AudioMidiDataset(TRAIN_AUDIO_PATH, TRAIN_MIDI_PATH, transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


TEMP_NUM_EPOCHS = 1
LEARNING_RATES = [0.001, 0.01, 0.005]
best_model = None
best_acc = float("-inf")

# for epoch in range(NUM_EPOCHS):
for epoch in range(TEMP_NUM_EPOCHS):
    for lr in LEARNING_RATES:
        model = AudioToMidiCNN()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
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

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {curr_loss / len(train_loader):.4f}"
        )
        test_dataset = AudioMidiDataset(TEST_AUDIO_PATH, TEST_MIDI_PATH, transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        results = evaluate_model(model, test_loader)
        acc = results["accuracy"]
        if acc > best_acc:
            best_acc = acc
            best_model = model


torch.save(best_model.state_dict(), MODEL_PATH)
print("Model saved successfully.")
