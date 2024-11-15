import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import os
import pretty_midi
import numpy as np


class Transformer:
    SAMPLE_RATE = 2000
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 20000
    MAX_AUDIO_LENGTH = 4917812
    MAX_MIDI_LENGTH = 245793

    @staticmethod
    def audio_to_mfcc(file_path: str, n_mfcc=13, hop_length=512):
        waveform, sr = torchaudio.load(file_path)

        if sr != Transformer.SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, sr, Transformer.SAMPLE_RATE
            )

        if waveform.shape[-1] < Transformer.MAX_AUDIO_LENGTH:
            pad_amount = Transformer.MAX_AUDIO_LENGTH - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad_amount), mode="constant", value=0)

        # Define the MFCC transform
        mfcc_transform = T.MFCC(
            sample_rate=Transformer.SAMPLE_RATE,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 2048, "hop_length": hop_length, "n_mels": 128},
        )
        # Convert the waveform to MFCCs
        mfcc = mfcc_transform(waveform)
        return mfcc

    @staticmethod
    def audio_to_mel_spec(
        audio_path,
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
    ):
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0).unsqueeze(dim=0)

        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

        if waveform.shape[-1] < Transformer.MAX_AUDIO_LENGTH:
            pad_amount = Transformer.MAX_AUDIO_LENGTH - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad_amount), mode="constant", value=0)

        mel_spec = T.MelSpectrogram(sample_rate, n_fft, n_mels, hop_length)(waveform)
        return mel_spec

    @staticmethod
    def midi_to_piano_roll(midi_path, fs=100):
        midi = pretty_midi.PrettyMIDI(midi_path)
        piano_roll = midi.get_piano_roll(fs=fs)
        piano_roll = (piano_roll > 0).astype(np.float32)
        piano_roll = torch.Tensor(piano_roll)
        if piano_roll.shape[-1] < Transformer.MAX_MIDI_LENGTH:
            pad_amount = Transformer.MAX_MIDI_LENGTH - piano_roll.shape[-1]
            piano_roll = F.pad(piano_roll, (0, pad_amount), mode="constant", value=0)
        return piano_roll

    @staticmethod
    def model_output_to_piano_roll(output, threshold=0.1):
        binary_output = torch.Tensor(output > threshold)
        piano_rolls = []
        for batch in range(binary_output.shape[0]):
            piano_roll = torch.zeros((128, binary_output.shape[2]))
            for i in range(128):
                piano_roll[i, :] = binary_output[batch, i, :]
            piano_rolls.append(piano_roll)
        return piano_rolls

    @staticmethod
    def maximum_audio_length(audio_dirs: list[str]):
        max_length = 0
        for audio_dir in audio_dirs:
            for file in os.listdir(audio_dir):
                file_path = os.path.join(audio_dir, file)
                waveform, sr = torchaudio.load(file_path)
                if sr != Transformer.SAMPLE_RATE:
                    waveform = torchaudio.transforms.Resample(
                        sr, Transformer.SAMPLE_RATE
                    )(waveform)
                max_length = max(max_length, waveform.shape[1])
        return max_length

    @staticmethod
    def maximum_midi_length(midi_dirs: list[str]):
        max_length = 0
        for midi_dir in midi_dirs:
            for file in os.listdir(midi_dir):
                midi_path = os.path.join(midi_dir, file)

                midi = pretty_midi.PrettyMIDI(midi_path)
                piano_roll = midi.get_piano_roll()
                max_length = max(max_length, piano_roll.shape[1])
        return max_length


if __name__ == "__main__":
    # max_length = Transformer.maximum_audio_length(
    #     ["data/train/audio", "data/test/audio"]
    # )
    # print(max_length)
    Transformer.audio_to_mel_spec("data/train/audio/00_BN1-129-Eb_comp_hex.wav")
    # Transformer.midi_to_piano_roll("data/train/midi/00_BN1-129-Eb_comp.mid")
    # max_length = Transformer.maximum_midi_length(["data/train/midi", "data/test/midi"])
    # print(max_length)
