import torch
import torch.nn as nn

class TaikoModel(nn.Module):
    def __init__(self, input_channels=1, n_mels=128, hidden_size=256, num_layers=2, output_dim=48):
        super(TaikoModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )

        self.freq_dim = n_mels // 4

        self.rnn = nn.LSTM(
            input_size=64 * self.freq_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, output_dim)

    def forward(self, x):
        batch_size, _, _, time_steps = x.size()
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch_size, x.size(1), -1)
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out)
        return out
