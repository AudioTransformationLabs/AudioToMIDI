import pretty_midi
import matplotlib.pyplot as plt
import librosa

from data_processing.transformer import Transformer

def plot_midi_annotation(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    plt.figure(figsize=(15, 6))

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start_time = note.start
            end_time = note.end
            pitch = note.pitch

            plt.plot(
                [start_time, end_time],
                [pitch, pitch],
                label=f"Note {pitch}",
                color="b",
                lw=2,
            )

    plt.xlabel("Time (seconds)")
    plt.ylabel("MIDI Pitch")
    plt.title("MIDI Notes Visualization")
    plt.show()


def plot_piano_roll(piano_roll, path: str = "results/piano_roll_output.png"):
    plt.figure(figsize=(15, 6))
    plt.imshow(piano_roll, aspect="auto", origin="lower", cmap="gray_r")
    plt.xlabel("Time (frames)")
    plt.ylabel("MIDI Pitch")
    plt.title("Piano Roll Visualization")
    plt.savefig(path)

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None, path: str = "results/spectrogram_output.png"):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
    plt.savefig(path)


# Example usage
# plot_piano_roll(Transformer.midi_to_piano_roll('data/test/midi/00_BN3-154-E_comp.mid'))
# plot_spectrogram(Transformer.transform_audio('data/test/audio/00_BN3-154-E_comp_hex.wav', Transformer.mfcc_transform())[0], path="results/mfcc_spectrogram_test.png")
# plot_spectrogram(Transformer.transform_audio('data/test/audio/00_BN3-154-E_comp_hex.wav', Transformer.mel_spec_transform())[0], path="results/mel_spectrogram_test.png")

