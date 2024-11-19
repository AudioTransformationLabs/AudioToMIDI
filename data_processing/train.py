import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

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

test_dataset = AudioMidiDataset(TEST_AUDIO_PATH, TEST_MIDI_PATH, transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

TEMP_NUM_EPOCHS = 1
LEARNING_RATES = [0.001, 0.01, 0.1]
CHUNK_LENGTHS = [512, 1024, 2048]
DROPOUT_PROBS = [1, 0.5, 0.3, 0.1]
best_model = None
best_acc = float("-inf")


for epoch in range(TEMP_NUM_EPOCHS):
    # for epoch in range(NUM_EPOCHS):
    for lr in LEARNING_RATES:
        for dropout in DROPOUT_PROBS:
            model = AudioToMidiCNN(dropout_prob=dropout)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            model.train()
            curr_loss = 0.0
            for features, labels in tqdm(
                train_loader,
                desc=f"Learning Rate: {lr}",
                total=len(train_loader),
            ):
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                curr_loss += loss.item()

                print(
                    f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {curr_loss / len(train_loader):.4f}"
                )

            results = evaluate_model(model, test_loader, device)
            acc = results["accuracy"]
            if acc > best_acc:
                best_acc = acc
                best_model = {"lr": lr, "dropout": dropout}
            torch.save(model.state_dict(), f"{MODEL_PATH}_dp={dropout}_lr={lr}")

with open("best_model_params.json", "w") as bmf:
    json.dump(best_model, bmf)
    print("Model saved successfully.")
