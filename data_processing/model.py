from torch import nn
import torch.nn.functional as F

BATCH_SIZE = 6


class AudioToMIDICNN(nn.Module):
    def __init__(self, time_frames):
        super(AudioToMIDICNN, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 8, kernel_size=(3, 3), padding=1
        )  # First layer with 8 filters
        self.conv2 = nn.Conv2d(
            8, 16, kernel_size=(3, 3), padding=1
        )  # Second layer with 16 filters

        # Pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d((2, 2))  # Reduce the dimensions by half each time

        # Another convolutional layer (last one)
        self.conv3 = nn.Conv2d(
            16, 32, kernel_size=(3, 3), padding=1
        )  # Third layer with 32 filters

        # Flatten and fully connected layer to map to MIDI output
        # self.fc1 = nn.Linear(
        #     32 * (time_frames) * 2, 128 * time_frames
        # )  # Adjust dimensions as needed

        # Output layer (reshape for 128 note probability over time)
        self.fc2 = nn.Linear(15360, 128 * time_frames)

    def forward(self, x):
        # Convolution layers with activation and pooling
        x = self.pool(F.relu(self.conv1(x)))  # First conv+pool
        x = self.pool(F.relu(self.conv2(x)))  # Second conv+pool
        x = self.pool(F.relu(self.conv3(x)))  # Third conv+pool

        # Flatten for the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layers
        # x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Final output

        # Reshape to (batch_size, 128, time_frames) for MIDI note prediction
        x = x.view(BATCH_SIZE, 128, -1)
        x = F.sigmoid(x)
        return x
