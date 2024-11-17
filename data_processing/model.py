import torch.nn as nn

class AudioToMidiCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=1024, dropout_prob=0.3):
        super(AudioToMidiCNN, self).__init__()

        # Define (Conv2D + BatchNorm2D + ReLU + MaxPool2D) blocks
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Time and frequency dims are divided by 2

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Time and frequency dims are divided by 2

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Time and frequency dims are divided by 2
        )

        # BiLSTM layer
        self.lstm_input_size = 2048
        self.bi_lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=2048,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)

        # Fully connected layer for output
        self.fc = nn.Linear(2048 * 2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input Shape: (batch_size, 1, 128, 1024)
        x = self.conv_block(x)  # Shape: (batch_size, 128, freq_bins, time_frames)

        batch_size, _, _, time_frames = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch_size, time_frames, -1)  # Shape: (batch_size, time_frames, features)

        x, _ = self.bi_lstm(x)  # Shape: (batch_size, time_frames, 2048)

        x = self.dropout(x)

        x = self.sigmoid(self.fc(x))  # Shape: (batch_size, time_frames, num_classes)

        return x
