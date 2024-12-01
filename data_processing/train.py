import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import (
    BATCH_SIZE,
    CHUNK_LENGTH,
    DEVICE,
    DROPOUT,
    FEATURE_TYPE,
    LEARNING_RATE,
    NUM_EPOCHS,
    TEST_AUDIO_PATH,
    TEST_MEL_SPEC_DATA_PATH,
    TEST_MFCC_DATA_PATH,
    TEST_MIDI_PATH,
    TRAIN_AUDIO_PATH,
    TRAIN_MEL_SPEC_DATA_PATH,
    TRAIN_MFCC_DATA_PATH,
    TRAIN_MIDI_PATH,
)
from .dataset import AudioMidiDataset
from .evaluate import evaluate_model, load_model
from .model import AudioToMidiCNN
from .transform import mel_spec_transform, mfcc_transform, get_model_path, get_optimizer_path

def load_checkpoint_if_exists(model_path, optimizer_path, params):
    if not os.path.exists(model_path) or not os.path.exists(optimizer_path):
        print(f"Model not found at {model_path}, creating new model.")
        model = AudioToMidiCNN(dropout=params["dropout"]).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        return model, optimizer
    else:
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        optimizer.load_state_dict(
            torch.load(optimizer_path, map_location=DEVICE, weights_only=True)
        )
        return model, optimizer


def load_dataset_if_exists(train_data_path, test_data_path):
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        transform = (
            mel_spec_transform()
            if FEATURE_TYPE == "mel_spec"
            else mfcc_transform()
        )
        train_dataset = AudioMidiDataset(TRAIN_AUDIO_PATH, TRAIN_MIDI_PATH, transform)
        torch.save(train_dataset, train_data_path)
        print("Train dataset saved successfully.")
        test_dataset = AudioMidiDataset(TEST_AUDIO_PATH, TEST_MIDI_PATH, transform)
        torch.save(test_dataset, test_data_path)
        print("Test dataset saved successfully.")
        return train_dataset, test_dataset
    else:
        train_dataset = torch.load(train_data_path, map_location=DEVICE)
        test_dataset = torch.load(test_data_path, map_location=DEVICE)
        print("Loaded train and test datasets successfully.")
        return train_dataset, test_dataset


def train_model(
    model,
    learning_rate,
    dropout,
    dataloader,
    optimizer,
    start_epoch=0,
):
    print(f"Training with Learning Rate: {learning_rate}, Dropout: {dropout}")
    model.train()
    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        curr_loss = 0.0
        for audio, midi in tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1} / {start_epoch + NUM_EPOCHS}",
            total=len(dataloader),
        ):
            audio, midi = audio.to(DEVICE), midi.to(DEVICE)
            optimizer.zero_grad()

            pos_samples = midi.sum(dim=(0, 2))
            neg_samples = (BATCH_SIZE * CHUNK_LENGTH) - pos_samples
            pos_weight = neg_samples / (pos_samples + 1e-8)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.unsqueeze(1))

            outputs = model(audio)

            loss = criterion(outputs, midi)
            loss.backward()
            optimizer.step()

            curr_loss += loss.item()

        print(
            f"Loss: {curr_loss / len(dataloader)}"
        )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Audio to MIDI model")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate for training")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="Dropout rate for the model")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of epochs for training")
    parser.add_argument("--feature_type", type=str, default=FEATURE_TYPE, choices=["mel_spec", "mfcc"], help="Feature type for audio transformation")
    parser.add_argument("--train_audio_path", type=str, default=TRAIN_AUDIO_PATH, help="Path to training audio files")
    parser.add_argument("--train_midi_path", type=str, default=TRAIN_MIDI_PATH, help="Path to training MIDI files")
    parser.add_argument("--test_audio_path", type=str, default=TEST_AUDIO_PATH, help="Path to testing audio files")
    parser.add_argument("--test_midi_path", type=str, default=TEST_MIDI_PATH, help="Path to testing MIDI files")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model file to load")
    parser.add_argument("--optimizer_path", type=str, default=None, help="Path to the optimizer file to load")
    parser.add_argument("--train_mel_spec_data_path", type=str, default=TRAIN_MEL_SPEC_DATA_PATH, help="Path to the training mel spectrogram dataset")
    parser.add_argument("--train_mfcc_data_path", type=str, default=TRAIN_MFCC_DATA_PATH, help="Path to the training MFCC dataset")
    parser.add_argument("--test_mel_spec_data_path", type=str, default=TEST_MEL_SPEC_DATA_PATH, help="Path to the testing mel spectrogram dataset")
    parser.add_argument("--test_mfcc_data_path", type=str, default=TEST_MFCC_DATA_PATH, help="Path to the testing MFCC dataset")

    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    FEATURE_TYPE = args.feature_type
    TRAIN_AUDIO_PATH = args.train_audio_path
    TRAIN_MIDI_PATH = args.train_midi_path
    TEST_AUDIO_PATH = args.test_audio_path
    TEST_MIDI_PATH = args.test_midi_path
    TRAIN_MEL_SPEC_DATA_PATH = args.train_mel_spec_data_path
    TRAIN_MFCC_DATA_PATH = args.train_mfcc_data_path
    TEST_MEL_SPEC_DATA_PATH = args.test_mel_spec_data_path
    TEST_MFCC_DATA_PATH = args.test_mfcc_data_path
    MODEL_PATH = args.model_path or get_model_path(FEATURE_TYPE, args.learning_rate, args.dropout)
    OPTIMIZER_PATH = args.optimizer_path or get_optimizer_path(FEATURE_TYPE, args.learning_rate, args.dropout)

    LEARNING_RATES = [args.learning_rate]
    DROPOUTS = [args.dropout]

    transform = (
        mel_spec_transform()
        if FEATURE_TYPE == "mel_spec"
        else mfcc_transform()
    )

    start_epoch = 0
    with open("./metadata.json", "r") as f:
        start_epoch = json.load(f)[FEATURE_TYPE]["epochs"]

    print(f"Using {DEVICE} to run model training")

    train_dataset, test_dataset = load_dataset_if_exists(
        TRAIN_MEL_SPEC_DATA_PATH if FEATURE_TYPE == "mel_spec" else TRAIN_MFCC_DATA_PATH,
        TEST_MEL_SPEC_DATA_PATH if FEATURE_TYPE == "mel_spec" else TEST_MFCC_DATA_PATH,
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_model = {}
    best_f1 = float("-inf")

    for learning_rate in LEARNING_RATES:
        for dropout in DROPOUTS:
            model, optimizer = load_checkpoint_if_exists(
                MODEL_PATH,
                OPTIMIZER_PATH,
                { "learning_rate": learning_rate, "dropout": dropout },
            )

            trained_model = train_model(
                model,
                learning_rate,
                dropout,
                train_loader,
                optimizer,
                start_epoch=start_epoch
            )
            torch.save(
                trained_model.state_dict(),
                MODEL_PATH,
            )
            torch.save(
                optimizer.state_dict(),
                OPTIMIZER_PATH,
            )

            results = evaluate_model(trained_model, test_loader)
            print(f"Results for LR={learning_rate}, Dropout={dropout}: {results}")

            if results["f1_score"] >= best_f1:
                best_f1 = results["f1_score"]
                best_model = {
                    "learning_rate": learning_rate,
                    "dropout": dropout,
                    "f1_score": best_f1,
                }

    with open("best_model_params.json", "w") as bmf:
        json.dump(best_model, bmf)
        print("Best model parameters saved successfully.")
