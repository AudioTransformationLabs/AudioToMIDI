import torch
from torch.utils.data import DataLoader
from data_processing.model import AudioToMidiCNN
from data_processing.constants import MODEL_NAME, TEST_AUDIO_PATH, TEST_MIDI_PATH, BATCH_SIZE
from data_processing.dataset import AudioMidiDataset
from data_processing.transformer import Transformer
from data_processing.visualization import plot_piano_roll

TEST_MODEL_PATH = f"{MODEL_NAME}_dp=0.3_lr=0.0005.pth"

def load_model(model_path):
    model = AudioToMidiCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

if __name__ == "__main__":
    transform = Transformer.mel_spec_transform()
    test_dataset = AudioMidiDataset(TEST_AUDIO_PATH, TEST_MIDI_PATH, transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(TEST_MODEL_PATH).to(device)

    # Run the model on one example audio
    example_audio, example_midi = next(iter(test_loader))
    example_audio = example_audio.to(device)
    with torch.no_grad():
        output = model(example_audio)
    
    print("Model Output Shape:", output.shape)
    plot_piano_roll(output[0].cpu().numpy(), path="model_output.png")
    plot_piano_roll(example_midi[0].cpu().numpy(), path="midi_annotation.png")
