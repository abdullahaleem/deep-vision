import torch
import torch.nn as nn
import torch.nn.functional as F


class AleemNet(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(6, momentum=None)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(16, momentum=None)
        self.fc1 = nn.Linear(16 * 56 * 56, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_features)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))    
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def aleemnet(in_channels, out_features):
    model = AleemNet(in_channels, out_features)
    return _cuda_enabled(model)


def _cuda_enabled(model):
    if torch.cuda.is_available():
        model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model