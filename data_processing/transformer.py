import torch
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

MAX_LENGTH = 39342495

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
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
    ):
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)

        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

        if waveform.shape[-1] < MAX_LENGTH:
            pad_amount = MAX_LENGTH - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad_amount), mode="constant", value=0)

        mel_spec = T.MelSpectrogram(sample_rate, n_fft, n_mels, hop_length)(waveform)
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
    def midi_to_piano_roll(midi_path, fs=100):
        midi = pretty_midi.PrettyMIDI(midi_path)
        piano_roll = midi.get_piano_roll(fs=fs)
        piano_roll = (piano_roll > 0).astype(np.float32)
        return piano_roll
    
    @staticmethod
    def model_output_to_piano_roll(output, threshold=0.1):
        binary_output = (output > threshold).numpy()
    
        piano_rolls = []
        for batch in range(binary_output.shape[0]):
            piano_roll = np.zeros((128, binary_output.shape[2]))
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
                if sr != 16000:
                    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                max_length = max(max_length, waveform.shape[1])
        return max_length