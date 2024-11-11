import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import os
import pretty_midi
import numpy as np
import sys

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
DURATION = 5

np.set_printoptions(threshold=sys.maxsize)

class Transformer:

    @staticmethod
    def audio_to_mfcc(audio_path: str, n_mfcc=13, sample_rate=22050, hop_length=512):
        # Load the audio file
        waveform, sr = torchaudio.load(audio_path)

        # Ensure the sample rate matches the expected one
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

        # Define the MFCC transform
        mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 2048, "hop_length": hop_length, "n_mels": 128},
        )

        # Convert the waveform to MFCCs
        mfcc = mfcc_transform(waveform)
        return mfcc

    @staticmethod
    def audio_to_mel_spec(
        audio_path,
        max_length=128,
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
    ):
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]

        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

        mel_spec = T.MelSpectrogram(sample_rate, n_fft, n_mels, hop_length)(waveform)

        if mel_spec.shape[-1] < max_length:
            pad_amount = max_length - mel_spec.shape[-1]
            mel_spec = F.pad(mel_spec, (0, pad_amount), mode="constant", value=0)
        else:
            mel_spec = mel_spec[:, :, :max_length]

        return mel_spec

    @staticmethod
    def convert_split_audio_to_mfcc(
        split_audio_path: str, n_mfcc=13, sample_rate=22050, hop_length=512
    ):
        results = []
        files = os.listdir(split_audio_path)
        for file in files:
            file_path = os.path.join(split_audio_path, file)
            results.append(
                Transformer.audio_to_mfcc(
                    file_path,
                    n_mfcc=n_mfcc,
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                )
            )
        return results

    @staticmethod
    def convert_split_audio_to_mel_spec(
        split_audio_path: str,
        max_length=128,
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
    ):
        results = []
        files = os.listdir(split_audio_path)
        for file in files:
            file_path = os.path.join(split_audio_path, file)
            results.append(
                Transformer.audio_to_mel_spec(
                    file_path,
                    max_length=max_length,
                    sample_rate=sample_rate,
                    n_mels=n_mels,
                    hop_length=hop_length,
                    n_fft=n_fft,
                )
            )
        return results

    @staticmethod
    def midi_to_label(midi_path):
        midi_data = pretty_midi.PrettyMIDI(midi_path).get_piano_roll(fs=100).T.astype(np.float32)
        return midi_data