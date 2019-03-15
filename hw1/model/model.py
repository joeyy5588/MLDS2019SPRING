import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

"""
CNNx: mean that it contains x layers before max pooling
"""
class CNN1(BaseModel):
    def __init__(self, num_classes=10):
        super(CNN1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 7, padding = 3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size = 7, padding = 3),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(1568, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class CNN2(BaseModel):
    def __init__(self, num_classes=10):
        super(CNN2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size = 5, padding = 2),
            nn.Conv2d(8, 8, kernel_size = 5, padding = 2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 24, kernel_size = 5, padding = 2),
            nn.Conv2d(24, 32, kernel_size = 5, padding = 2),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(1568, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class CNN3(BaseModel):
    def __init__(self, num_classes=10):
        super(CNN3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 3, padding = 1),
            nn.Conv2d(16, 16, kernel_size = 3, padding = 1),
            nn.Conv2d(16, 16, kernel_size = 3, padding = 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
            nn.Conv2d(32, 32, kernel_size = 3, padding = 1),
            nn.Conv2d(32, 32, kernel_size = 3, padding = 1),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(1568, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class CNNh(BaseModel):
    def __init__(self, c1, c2, c3, c4, num_classes=10):
        super(CNNh, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, int(8 * c1), kernel_size = 5, padding = 2),
            nn.Conv2d(int(8 * c1), int(8 * c2), kernel_size = 5, padding = 2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(int(8 * c2), int(24 * c3), kernel_size = 5, padding = 2),
            nn.Conv2d(int(24 * c3), int(32 * c4), kernel_size = 5, padding = 2),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(int(7 * 7 * 32 * c4), 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class DNN2(BaseModel):
    def __init__(self):
        super(DNN2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 190),
            nn.ReLU(inplace=True),
            nn.Linear(190, 1),
        )

    def forward(self, x):
        output = self.fc(x)
        return output

class DNN5(BaseModel):
    def __init__(self):
        super(DNN5, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1),
        )
    def forward(self, x):
        output = self.fc(x)
        return output

class DNN8(BaseModel):
    def __init__(self):
        super(DNN8, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 5),
            nn.ReLU(inplace=True),
            nn.Linear(5, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 5),
            nn.ReLU(inplace=True),
            nn.Linear(5, 1)
        )

    def forward(self, x):
        output = self.fc(x)
        return output
