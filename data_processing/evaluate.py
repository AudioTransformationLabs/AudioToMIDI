import argparse
import os
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from .constants import (
    BATCH_SIZE,
    CHUNK_LENGTH,
    DEVICE,
    DROPOUT,
    FEATURE_TYPE,
    LEARNING_RATE,
    TEST_AUDIO_PATH,
    TEST_MIDI_PATH,
    THRESHOLD,
)
from .dataset import AudioMidiDataset
from .model import AudioToMidiCNN
from .transform import get_model_path, mel_spec_transform, mfcc_transform

def load_model(model_path):
    model = AudioToMidiCNN()
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, creating new model.")
        return model.to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    return model.to(DEVICE)

def evaluate_model(model, dataloader):
    total_loss = 0.0
    precision, recall, f1 = 0.0, 0.0, 0.0

    with torch.no_grad():
        model.eval()
        for audio_chunks, midi_chunks in tqdm(
            dataloader, desc="Evaluating on test batches", total=len(dataloader)
        ):
            audio_chunks, midi_chunks = audio_chunks.to(DEVICE), midi_chunks.to(DEVICE)
            outputs = model(audio_chunks)

            pos_samples = midi_chunks.sum(dim=(0, 2))
            neg_samples = (BATCH_SIZE * CHUNK_LENGTH) - pos_samples
            pos_weight = neg_samples / (pos_samples + 1e-8)
            criterion = BCEWithLogitsLoss(pos_weight=pos_weight.unsqueeze(1))

            loss = criterion(outputs, midi_chunks)
            total_loss += loss.item()

            midi_chunks = midi_chunks.cpu().numpy().reshape(-1, 128)
            outputs = (torch.sigmoid(outputs) >= THRESHOLD).float().cpu().numpy().reshape(-1, 128)

            precision = precision_score(midi_chunks, outputs, zero_division=0, average="samples")
            recall = recall_score(midi_chunks, outputs, zero_division=0, average="samples")
            f1 = f1_score(midi_chunks, outputs, zero_division=0, average="samples")

    avg_loss = total_loss / len(dataloader)
    avg_precision = precision / len(dataloader)
    avg_recall = recall / len(dataloader)
    avg_f1 = f1 / len(dataloader)

    return {
        "loss": avg_loss,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_score": avg_f1,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model on the test dataset.")
    parser.add_argument(
        "--feature_type",
        type=str,
        default=FEATURE_TYPE,
        choices=["mel_spec", "mfcc"],
        help="Feature type used for training the model.",
    )
    parser.add_argument(
        "--test_audio_path",
        type=str,
        default=TEST_AUDIO_PATH,
        help="Path to the test audio dataset.",
    )
    parser.add_argument(
        "--test_midi_path",
        type=str,
        default=TEST_MIDI_PATH,
        help="Path to the test midi dataset.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size used during training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate used during training.",
    )
    parser.add_argument(
        "--dropout", type=float, default=DROPOUT, help="Dropout used during training."
    )
    args = parser.parse_args()

    FEATURE_TYPE = args.feature_type
    TEST_AUDIO_PATH = args.test_audio_path
    TEST_MIDI_PATH = args.test_midi_path
    BATCH_SIZE = args.batch_size

    LEARNING_RATE = args.learning_rate
    DROPOUT = args.dropout

    transform = (
        mel_spec_transform() if FEATURE_TYPE == "mel_spec" else mfcc_transform()
    )

    test_dataset = AudioMidiDataset(TEST_AUDIO_PATH, TEST_MIDI_PATH, transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = load_model(get_model_path(LEARNING_RATE, DROPOUT))

    results = evaluate_model(model, test_loader)
    print("Evaluation Metrics:", results)
