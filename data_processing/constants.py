# Audio Processing Parameters
SAMPLE_RATE = 44100
N_MELS = 128
N_FFT = 2048
N_MFCC = 128
HOP_LENGTH = 512
CHUNK_LENGTH = 1024

# Dataset Paths
TRAIN_AUDIO_PATH = "data/train/audio"
TRAIN_MIDI_PATH = "data/train/midi"
TEST_AUDIO_PATH = "data/test/audio"
TEST_MIDI_PATH = "data/test/midi"

# Model Filename
MODEL_NAME = "AudioToMidiCNN"

# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DROPOUT = 0.5
FEATURE_TYPE = "mel_spec"  # or "mfcc"
