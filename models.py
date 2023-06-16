import torch.nn as nn
import torch.nn.functional as F


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.mnist_model = nn.Sequential(
            nn.Conv2d(1, 8, 3, bias=False, padding=1),  # RF = 3
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1),
            nn.Conv2d(8, 8, 3, bias=False, padding=1),  # RF = 5
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1),
            nn.Conv2d(8, 8, 3, bias=False, padding=1),  # RF = 7
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1),
            nn.Conv2d(8, 8, 3, bias=False, padding=1),  # RF = 9
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),  # RF = 10
            nn.Dropout(0.25),
            nn.Conv2d(8, 16, 1, bias=False),  # RF = 14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
            nn.Conv2d(16, 16, 3, bias=False),  # RF = 18
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),  # RF = 20
            nn.Dropout(0.25),
            nn.Conv2d(16, 16, 3),  # RF = 28
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 10, 3),  # RF = 36
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        x = self.mnist_model(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)


class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mnist_classifier = nn.Sequential(
            nn.Conv2d(1, 8, 3, bias=False),  # RF = 3
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.05),
            nn.Conv2d(8, 16, 3, bias=False),  # RF = 5
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),  # RF = 6
            nn.Dropout(0.1),
            nn.Conv2d(16, 16, 3, bias=False),  # RF = 10
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
            nn.Conv2d(16, 16, 3, bias=False),  # RF = 14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),  # RF = 16
            nn.Dropout(0.1),
            nn.Conv2d(16, 10, 3, bias=False),  # RF = 24
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        x = self.mnist_classifier(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class Model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.mnist_classifier = nn.Sequential(
            nn.Conv2d(1, 8, 3, bias=False),  # RF = 3
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.05),
            nn.Conv2d(8, 16, 3, bias=False),  # RF = 5
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),  # RF = 6
            nn.Dropout(0.1),
            nn.Conv2d(16, 16, 3, bias=False),  # RF = 10
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
            nn.Conv2d(16, 16, 3, bias=False),  # RF = 14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),  # RF = 16
            nn.Dropout(0.1),
            nn.Conv2d(16, 10, 3, bias=False),  # RF = 24
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        x = self.mnist_classifier(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
