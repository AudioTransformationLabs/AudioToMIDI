import torch
from torch.utils.data import DataLoader

from .dataset import AudioMidiDataset
from .model import AudioToMidiCNN
from .constants import (
    FEATURE_TYPE,
    TEST_AUDIO_PATH,
    TEST_MIDI_PATH,
    BATCH_SIZE,
    MODEL_NAME,
)
from .transformer import Transformer


def load_model(model_path):
    model = AudioToMidiCNN()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def evaluate_model(model, dataloader, device, threshold=0.5):
    total_loss = 0

    correct = 0
    total = 0

    criterion = torch.nn.BCELoss()

    model.to(device)
    with torch.no_grad():
        for audio, midi in dataloader:
            audio, midi = audio.to(device), midi.to(device)
            outputs = model(
                audio
            )  # Shape: (batch_size, num_classes [128], num_frames [1024])

            loss = criterion(outputs, midi)
            total_loss += loss.item()

            predictions = (torch.sigmoid(outputs) > threshold).float()

            correct += (predictions == midi).sum().item()
            total += midi.numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return {"loss": avg_loss, "accuracy": accuracy}


if __name__ == "__main__":
    transform = (
        Transformer.mel_spec_transform()
        if FEATURE_TYPE == "mel_spec"
        else Transformer.mfcc_transform()
    )

    test_dataset = AudioMidiDataset(TEST_AUDIO_PATH, TEST_MIDI_PATH, transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_NAME).to(device)

    results = evaluate_model(model, test_loader, device)
    print("Evaluation Metrics:", results)
