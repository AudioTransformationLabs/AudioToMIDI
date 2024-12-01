import argparse
import os
import torch
from more_itertools import numeric_range
from .constants import (
    CHUNK_LENGTH,
    DEVICE,
    DROPOUT,
    HOP_LENGTH,
    LEARNING_RATE,
    N_MELS,
    TEST_AUDIO_PATH,
    TEST_MIDI_PATH
)
from .transform import (
    get_model_path,
    remove_short_fragments,
    fill_segment_gaps,
    mel_spec_transform,
    split_audio_midi_pair
)
from .visualization import plot_piano_roll, plot_spectrogram
from .evaluate import load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model on a sample audio file.")
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
        "--chunk_length",
        type=int,
        default=CHUNK_LENGTH,
        help="Length of the audio chunks used for testing.",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=HOP_LENGTH,
        help="Hop length used for splitting the audio and MIDI files.",
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

    CHUNK_LENGTH = args.chunk_length
    HOP_LENGTH = args.hop_length
    LEARNING_RATE = args.learning_rate
    DROPOUT = args.dropout
    TEST_AUDIO_PATH = args.test_audio_path
    TEST_MIDI_PATH = args.test_midi_path

    AUDIO_PATH = f"{TEST_AUDIO_PATH}/00_BN3-154-E_comp_hex.wav"
    MIDI_PATH = f"{TEST_MIDI_PATH}/00_BN3-154-E_comp.mid"

    transform = mel_spec_transform()

    audio_chunks, midi_chunks = split_audio_midi_pair(AUDIO_PATH, MIDI_PATH, transform, CHUNK_LENGTH, HOP_LENGTH)

    model_file = get_model_path(LEARNING_RATE, DROPOUT)
    model = load_model(model_file)
    audio, midi = audio_chunks[0], midi_chunks[0]
    actual_piano_roll = midi.cpu().numpy()
    with torch.no_grad():
        os.makedirs("results", exist_ok=True)
        plot_spectrogram(audio[0], title="Mel Spectrogram", path="results/mel_spectrogram.png")
        audio = audio.reshape(1, 1, 128, 1024).to(DEVICE)
        for t in numeric_range(0.1, 0.95, 0.1):
            predicted_piano_roll = (torch.sigmoid(model(audio)[0]) >= t).float()
            predicted_piano_roll = remove_short_fragments(fill_segment_gaps(predicted_piano_roll.cpu().numpy())).reshape(N_MELS, CHUNK_LENGTH)
            plot_piano_roll(predicted_piano_roll, path=f"results/predicted_piano_roll_{t:.1f}.png")

    plot_piano_roll(actual_piano_roll, path="results/actual_piano_roll.png")