from torch import nn
import torch

class AudioToMIDICNN(nn.Module):
    def __init__(self):
        super(AudioToMIDICNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.fc1 = nn.Linear(64 * 16 * 19210, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128 * 128)  # Output size for MIDI representation

    def forward(self, x):
        if x.dim() == 3:  # If input is (batch_size, height, width)
            x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = nn.ReLU()(self.fc3(x))
        x = self.fc4(x)
        x = x.view(x.size(0), 128, 128)  # Reshape to MIDI representation
        return x
