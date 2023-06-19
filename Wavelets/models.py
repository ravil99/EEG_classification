import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.conv0 = nn.Conv2d(3, 8, kernel_size=4, padding=2)
        self.relu0 = nn.ReLU()
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)

        self.conv1 = nn.Conv2d(8, 16, kernel_size=4, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=8)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(32 * 4 * 4, 240)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(240, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.maxpool0(x)
        

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.dense2(x)
        # x = self.softmax(x)
        return x
