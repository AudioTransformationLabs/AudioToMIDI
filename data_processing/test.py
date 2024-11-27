import torch
from torch.nn.functional import sigmoid
from data_processing.model import AudioToMidiCNN
from data_processing.constants import CHUNK_LENGTH, DROPOUT, HOP_LENGTH, LEARNING_RATE, TRAIN_AUDIO_PATH, TRAIN_MIDI_PATH
from data_processing.transformer import Transformer
from data_processing.visualization import plot_piano_roll, plot_spectrogram

def load_model(model_path):
    model = AudioToMidiCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

if __name__ == "__main__":
    AUDIO_PATH = f"{TRAIN_AUDIO_PATH}/00_BN1-129-Eb_comp_hex.wav"
    MIDI_PATH = f"{TRAIN_MIDI_PATH}/00_BN1-129-Eb_comp.mid"

    transform = Transformer.mel_spec_transform()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_chunks, midi_chunks = Transformer.split_audio_midi_pair(AUDIO_PATH, MIDI_PATH, transform, CHUNK_LENGTH, HOP_LENGTH)

    model = load_model(f"models/AudioToMidiCNN_melspec_LR=0.001_DROPOUT=0.5.pth")
    audio, midi = audio_chunks[0], midi_chunks[0]
    actual_piano_roll = midi.cpu().numpy()
    with torch.no_grad():
        plot_spectrogram(audio[0], title="Mel Spectrogram", path="results/mel_spectrogram.png")
        audio = audio.reshape(1, 1, 128, 1024).to(device)
        predicted_piano_roll = (sigmoid(model(audio)[0]) >= 0.5).float()

    plot_piano_roll(actual_piano_roll, path="results/actual_piano_roll.png")
    plot_piano_roll(predicted_piano_roll.cpu().numpy(), path="results/predicted_piano_roll.png")
