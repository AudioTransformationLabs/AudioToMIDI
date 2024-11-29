import os
import torch
from torch.nn.functional import sigmoid
from data_processing.constants import CHUNK_LENGTH, HOP_LENGTH, N_MELS, TEST_AUDIO_PATH, TEST_MIDI_PATH
from data_processing.transformer import Transformer
from data_processing.visualization import plot_piano_roll, plot_spectrogram
from data_processing.evaluate import load_model
from more_itertools import numeric_range

if __name__ == "__main__":
    AUDIO_PATH = f"{TEST_AUDIO_PATH}/00_BN3-154-E_comp_hex.wav"
    MIDI_PATH = f"{TEST_MIDI_PATH}/00_BN3-154-E_comp.mid"

    transform = Transformer.mel_spec_transform()
    device = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    audio_chunks, midi_chunks = Transformer.split_audio_midi_pair(AUDIO_PATH, MIDI_PATH, transform, CHUNK_LENGTH, HOP_LENGTH)

    model = load_model(f"models/AudioToMidiCNN_melspec_LR=0.001_DROPOUT=0.5.pth", device)
    audio, midi = audio_chunks[0], midi_chunks[0]
    actual_piano_roll = midi.cpu().numpy()
    with torch.no_grad():
        os.makedirs("results", exist_ok=True)
        plot_spectrogram(audio[0], title="Mel Spectrogram", path="results/mel_spectrogram.png")
        audio = audio.reshape(1, 1, 128, 1024).to(device)
        for t in numeric_range(0.5, 0.95, 0.05):
            predicted_piano_roll = (sigmoid(model(audio)[0]) >= t).float()
            predicted_piano_roll = Transformer.remove_short_fragments(Transformer.fill_segment_gaps(predicted_piano_roll.cpu().numpy().reshape(1, N_MELS, CHUNK_LENGTH))).reshape(N_MELS, CHUNK_LENGTH)
            plot_piano_roll(predicted_piano_roll, path=f"results/predicted_piano_roll_{t}.png")

    plot_piano_roll(actual_piano_roll, path="results/actual_piano_roll.png")
