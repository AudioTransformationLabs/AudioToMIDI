import torch
from torch.utils.data import DataLoader
from torch.nn.functional import sigmoid
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from .dataset import AudioMidiDataset
from .model import AudioToMidiCNN
from .constants import (
    CHUNK_LENGTH,
    DROPOUT,
    FEATURE_TYPE,
    LEARNING_RATE,
    TEST_AUDIO_PATH,
    TEST_MIDI_PATH,
    BATCH_SIZE,
    MODEL_NAME,
)
from .transformer import Transformer

def load_model(model_path, device):
    model = AudioToMidiCNN()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model.to(device)

def evaluate_model(model, dataloader, device, threshold=0.7):
    total_loss = 0.0
    precision, recall, f1 = 0.0, 0.0, 0.0

    with torch.no_grad():
        for audio_chunks, midi_chunks in tqdm(dataloader, desc="Evaluating on test batches", total=len(dataloader)):
            audio_chunks, midi_chunks = audio_chunks.to(device), midi_chunks.to(device)
            outputs = (sigmoid(model(audio_chunks)) >= threshold).float()
            outputs = torch.Tensor(Transformer.remove_short_fragments(
                Transformer.fill_segment_gaps(outputs.cpu().numpy())
            )).to(device)

            pos_samples = midi_chunks.sum(dim=(0, 2))
            neg_samples = (BATCH_SIZE * CHUNK_LENGTH) - pos_samples
            pos_weight = neg_samples / (pos_samples + 1e-8)
            criterion = BCEWithLogitsLoss(pos_weight=pos_weight.unsqueeze(1))

            loss = criterion(outputs, midi_chunks)
            total_loss += loss.item()

            midi_chunks = midi_chunks.cpu().numpy().reshape(-1, 128)
            outputs = outputs.cpu().numpy().reshape(-1, 128)

            precision = precision_score(midi_chunks, outputs, zero_division=0, average='samples')
            recall = recall_score(midi_chunks, outputs, zero_division=0, average='samples')
            f1 = f1_score(midi_chunks, outputs, zero_division=0, average='samples')

    avg_loss = total_loss / len(dataloader)
    avg_precision = precision / len(dataloader)
    avg_recall = recall / len(dataloader)
    avg_f1 = f1 / len(dataloader)

    return {
        "loss": avg_loss,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_score": avg_f1
    }


if __name__ == "__main__":
    transform = (
        Transformer.mel_spec_transform()
        if FEATURE_TYPE == "mel_spec"
        else Transformer.mfcc_transform()
    )

    test_dataset = AudioMidiDataset(TEST_AUDIO_PATH, TEST_MIDI_PATH, transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(f"models/{MODEL_NAME}_melspec_LR={LEARNING_RATE}_DROPOUT={DROPOUT}.pth", device)

    results = evaluate_model(model, test_loader, device)
    print("Evaluation Metrics:", results)
