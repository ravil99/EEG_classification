import torch
from torch import nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim,
                           batch_first=True, dropout=dropout_rate)
        self.rnn_2 = nn.LSTM(hidden_dim, 128, layer_dim,
                             batch_first=True, dropout=dropout_rate)
        self.rnn_3 = nn.LSTM(128, 64, layer_dim,
                             batch_first=True, dropout=dropout_rate)
        self.rnn_4 = nn.LSTM(64, 32, layer_dim,
                             batch_first=True, dropout=dropout_rate)
        self.rnn_5 = nn.LSTM(32, 16, layer_dim,
                             batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, 64)         # or 64
        self.fc_2 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layer_norm = nn.LayerNorm([hidden_dim])
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x, self.hidden_dim)
        out, (hn, cn) = self.rnn(x, (h0, c0))

        # h1, c1 = self.init_hidden(x, 128)
        # out, (hn, cn) = self.rnn_2(out, (h1, c1))
        # h2, c2 = self.init_hidden(x, 64)
        # out, (hn, cn) = self.rnn_3(out, (h2, c2))
        # h3, c3 = self.init_hidden(x, 32)
        # out, (hn, cn) = self.rnn_4(out, (h3, c3))
        # h4, c4 = self.init_hidden(x, 16)
        # out, (hn, cn) = self.rnn_5(out, (h4, c4))

        # out = self.layer_norm(out[:, -1, :])
        out = self.fc(out[:, -1, :])
        # НЕПРАВИЛЬНО считается функция ошибки
        # Use Sigmoid, LeakyReLu or TanH

        # out = F.tanh(out)
        # out = self.fc_2(out)

        return out

    def init_hidden(self, x, hidden_dim):
        # Инициализация НАЧАЛЬНОГО hidden состояния.
        # Можно 0, можно рандомом
        h0 = torch.zeros(self.layer_dim, x.size(0), hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), hidden_dim)
        return [t.cuda() for t in (h0, c0)]


class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=32),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        # self.layer_norm = nn.LayerNorm([64])
        self.lstm_2 = nn.LSTM(input_size=64, hidden_size=32,
                              num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        # self.layer_norm_2 = nn.LayerNorm([32])
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # cnn takes input of shape (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)
        out = self.cnn(x)
        # lstm takes input of shape (batch_size, seq_len, input_size)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm(out)
        # out, _ = self.lstm_2(out)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class LSTM_CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_CNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=32),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # flatten
            nn.Flatten(),
            nn.LazyLinear(out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.permute(0, 2, 1)
        out = self.cnn(out)
        return out


class CNBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(CNBlock, self).__init__()

        self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.kernel_size = kernel_size
        # self.stride = stride
        # self.padding = padding

        self.conv1 = nn.Conv1d(in_channels=self.in_channels,
                               out_channels=32, kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=self.in_channels,
                               out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=self.in_channels,
                               out_channels=32, kernel_size=8, stride=2, padding=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class ChronoNet(nn.Module):
    def __init__(self, first_gru_length, out_dim=8, input_dim=11):
        super().__init__()

        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.kernel_size = kernel_size
        # self.stride = stride
        # self.padding = padding

        self.block1 = CNBlock(input_dim)
        self.block2 = CNBlock(96)
        self.block3 = CNBlock(96)

        self.gru1 = nn.GRU(input_size=96, hidden_size=32, batch_first=True)
        self.gru2 = nn.GRU(input_size=32, hidden_size=32, batch_first=True)
        self.gru3 = nn.GRU(input_size=64, hidden_size=32, batch_first=True)
        self.gru4 = nn.GRU(input_size=96, hidden_size=32, batch_first=True)
        self.gru_linear = nn.Linear(first_gru_length, 1)
        self.flatten = nn.Flatten()
        self.fcl = nn.Linear(32, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.permute(0, 2, 1)

        gru_out1, _ = self.gru1(x)
        gru_out2, _ = self.gru2(gru_out1)
        gru_out = torch.cat([gru_out1, gru_out2], dim=2)
        gru_out3, _ = self.gru3(gru_out)
        gru_out = torch.cat([gru_out1, gru_out2, gru_out3], dim=2)

        linear_out = self.relu(self.gru_linear(gru_out.permute(0, 2, 1)))
        gru_out4, _ = self.gru4(linear_out.permute(0, 2, 1))
        x = self.flatten(gru_out4)
        x = self.fcl(x)
        return x
