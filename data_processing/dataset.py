import os
from torch.utils.data import Dataset
from .transformer import Transformer

# import torchaudio


SAMPLE_RATE = Transformer.SAMPLE_RATE


class AudioMidiDataset(Dataset):
    def __init__(self, audio_path, midi_path, feature_type):
        self.audio_files = sorted(
            [os.path.join(audio_path, f) for f in os.listdir(audio_path)]
        )
        self.midi_files = sorted(
            [
                os.path.join(midi_path, f.replace(".midi", ".mid"))
                for f in os.listdir(midi_path)
            ]
        )
        self.feature_type = feature_type
        self.all_features = []

    # def split_path_to_multiple_instances(self, file_path: str):
    #     path_features = Transformer.audio_to_mfcc
    #
    #     waveform, sr = torchaudio.load(file_path)
    #     print(waveform)
    #     print(waveform.shape)
    #
    #     if sr != SAMPLE_RATE:
    #         waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_filepath = self.audio_files[idx]
        midi_filepath = self.midi_files[idx]
        if self.feature_type == "mfcc":
            features = Transformer.audio_to_mfcc(audio_filepath)
        else:
            features = Transformer.audio_to_mel_spec(audio_filepath)

        midi_label = Transformer.midi_to_piano_roll(midi_filepath)
        return features, midi_label
