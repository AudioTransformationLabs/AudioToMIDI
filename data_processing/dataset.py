import os
from torch.utils.data import Dataset
from tqdm import tqdm

from .constants import CHUNK_LENGTH, HOP_LENGTH
from .transformer import Transformer


class AudioMidiDataset(Dataset):
    def __init__(self, audio_path, midi_path, transform):
        self.audio_files = sorted(
            [os.path.join(audio_path, f) for f in os.listdir(audio_path)]
        )
        self.midi_files = sorted(
            [os.path.join(midi_path, f) for f in os.listdir(midi_path)]
        )
        self.transform = transform

        self.data = []
        for audio_file, midi_file in tqdm(
            zip(self.audio_files, self.midi_files),
            desc="Processing audio-midi pairs",
            total=len(self.audio_files),
        ):
            audio_chunks, midi_chunks = Transformer.split_audio_midi_pair(
                audio_file, midi_file, self.transform, CHUNK_LENGTH, HOP_LENGTH
            )
            self.data.extend(list(zip(audio_chunks, midi_chunks)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
