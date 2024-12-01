import torch

# Audio Processing Parameters
SAMPLE_RATE = 44100
N_MELS = 128
N_FFT = 2048
N_MFCC = 13
HOP_LENGTH = 512
CHUNK_LENGTH = 1024

# Dataset Paths
TRAIN_AUDIO_PATH = "data/train/audio"
TRAIN_MIDI_PATH = "data/train/midi"
TEST_AUDIO_PATH = "data/test/audio"
TEST_MIDI_PATH = "data/test/midi"

TRAIN_MEL_SPEC_DATA_PATH = f"data/mel_spec_train_dataset.pth"
TRAIN_MFCC_DATA_PATH = f"data/mfcc_train_dataset.pth"
TEST_MEL_SPEC_DATA_PATH = f"data/mel_spec_test_dataset.pth"
TEST_MFCC_DATA_PATH = f"data/mfcc_test_dataset.pth"

# Model Filename
MODEL_NAME = "AudioToMidiCNN"

# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 25
DROPOUT = 0.5
FEATURE_TYPE = "mel_spec"
# FEATURE_TYPE = "mfcc"
THRESHOLD = 0.7
DEVICE = "mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
