import os
import torch
from torch.utils.data import Dataset, DataLoader

from .visualization import plot_midi_annotation, plot_piano_roll
from .transformer import Transformer
from .model import AudioToMIDICNN
from tqdm import tqdm

class AudioMidiDataset(Dataset):
    def __init__(self, audio_path, midi_path, feature_type):
        self.audio_files = sorted([
            os.path.join(audio_path, f) for f in os.listdir(audio_path)
        ])
        self.midi_files = sorted([
            os.path.join(midi_path, f.replace('.midi', '.mid')) for f in os.listdir(midi_path)
        ])
        self.feature_type = feature_type

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_filepath = self.audio_files[idx]
        midi_filepath = self.midi_files[idx]

        if self.feature_type == 'mfcc':
            features = Transformer.audio_to_mfcc(audio_filepath)
        else:
            features = Transformer.audio_to_mel_spec(audio_filepath)

        midi_label = Transformer.midi_to_piano_roll(midi_filepath)
        
        return features, midi_label

train_audio_dir = "data/train/audio"
train_midi_dir = "data/train/midi"
test_audio_dir = "data/test/audio"
test_midi_dir = "data/test/midi"

# train_dataset = AudioMidiDataset(train_audio_dir, train_midi_dir, feature_type='mel_spectrogram')
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_dataset = AudioMidiDataset(test_audio_dir, test_midi_dir, feature_type='mel_spectrogram')
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = AudioToMIDICNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = torch.nn.BCELoss()

# for batch_idx, (features, midi_data) in tqdm(enumerate(test_loader)):
#     print(f"Batch {batch_idx + 1}")
#     print("Features shape:", features.shape)  # (batch, n_mels, time)
#     print("MIDI data shape:", midi_data.shape)  # Shape based on how you process MIDI files

features = Transformer.audio_to_mel_spec('data/test/audio/00_BN3-154-E_comp_hex.wav')
features = torch.tensor(features).unsqueeze(0)
print(features.shape)
optimizer.zero_grad()
output = model(features)
print(output)
plot_piano_roll(Transformer.model_output_to_piano_roll(output)[0])
# loss = criterion(output, torch.zeros_like(output))
# loss.backward()
# optimizer.step()

    
