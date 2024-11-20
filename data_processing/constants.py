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

# Model Path
MODEL_NAME = "audio_to_midi_cnn"

# Training Parameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
FEATURE_TYPE = "mel_spec"  # or "mfcc"

