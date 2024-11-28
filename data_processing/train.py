import os
import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import (
    BATCH_SIZE,
    CHUNK_LENGTH,
    FEATURE_TYPE,
    LEARNING_RATE,
    MODEL_NAME,
    NUM_EPOCHS,
    TRAIN_AUDIO_PATH,
    TRAIN_MIDI_PATH,
    TEST_AUDIO_PATH,
    TEST_MIDI_PATH,
)
from .dataset import AudioMidiDataset
from .evaluate import evaluate_model, load_model
from .model import AudioToMidiCNN
from .transformer import Transformer

def load_checkpoint_if_exists(model_path, optimizer_path, params, device):
    if not os.path.exists(model_path) or not os.path.exists(optimizer_path):
        model = AudioToMidiCNN(dropout=params['dropout']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        return model, optimizer
    else:
        model = load_model(model_path, device)
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))
        return model, optimizer

def train_model(model, learning_rate, dropout, dataloader, optimizer, device, start_epoch=0, epochs=NUM_EPOCHS, threshold=0.7):
    print(f'Learning Rate: {learning_rate}, Dropout: {dropout}')
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        curr_loss = 0.0
        for audio, midi in tqdm(dataloader, desc=f"Epoch {epoch + 1} / {start_epoch + epochs}", total=len(dataloader)):
            audio, midi = audio.to(device), midi.to(device)
            optimizer.zero_grad()

            pos_samples = midi.sum(dim=(0, 2))
            neg_samples = (BATCH_SIZE * CHUNK_LENGTH) - pos_samples
            pos_weight = neg_samples / (pos_samples + 1e-8)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.unsqueeze(1))

            outputs = model(audio)
            outputs = (torch.sigmoid(model(audio)) >= threshold).float()
            outputs = torch.Tensor(Transformer.remove_short_fragments(
                Transformer.fill_segment_gaps(outputs.cpu().numpy())
            )).to(device)

            loss = criterion(outputs, midi)
            loss.backward()
            optimizer.step()

            curr_loss += loss.item()

        print(f"Epoch {epoch + 1}/{start_epoch + epochs}, Loss: {curr_loss / len(dataloader)}")

    return model

if __name__ == "__main__":
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

    LEARNING_RATES = [0.0001, 0.001, 0.01]
    DROPOUTS = [1, 0.5, 0.1]
    best_model = {}
    best_f1 = float("-inf")

    for learning_rate in LEARNING_RATES:
        for dropout in DROPOUTS:
            model, optimizer = load_checkpoint_if_exists(
                f"models/{MODEL_NAME}_melspec_LR={learning_rate}_DROPOUT={dropout}.pth",
                f"models/{MODEL_NAME}_melspec_LR={learning_rate}_DROPOUT={dropout}_optimizer.pth",
                { "learning_rate": learning_rate, "dropout": dropout },
                device
            )

            # TODO: Obtain start_epoch from model checkpoint state dict
            trained_model = train_model(
                model, learning_rate, dropout, train_loader, optimizer, start_epoch=100, epochs=NUM_EPOCHS
            )
            torch.save(trained_model.state_dict(), f"models/{MODEL_NAME}_melspec_LR={learning_rate}_DROPOUT={dropout}.pth")
            torch.save(optimizer.state_dict(), f"models/{MODEL_NAME}_melspec_LR={learning_rate}_DROPOUT={dropout}_optimizer.pth")

            results = evaluate_model(trained_model, test_loader, device)
            print(f"Results for LR={learning_rate}, Dropout={dropout}: {results}")

            if results['f1_score'] >= best_f1:
                best_f1 = results['f1_score']
                best_model = {
                    "learning_rate": learning_rate,
                    "dropout": dropout,
                    "f1_score": best_f1,
                }

    with open("best_model_params.json", "w") as bmf:
        json.dump(best_model, bmf)
        print("Best model parameters saved successfully.")
