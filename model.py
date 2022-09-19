import torch
import torch.nn as nn
import torch.nn.functional as F

class PGNet(nn.Module):
    def __init__(self, input_dim, action_size, dropout):
        self.in_channels, self.height, self.width= input_dim
        self.action_size = action_size
        self.dropout = dropout

        super(PGNet, self).__init__()
        self.conv1 = nn.Conv2d(self.in_channels, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64*(self.height//8)*(self.width//8), 256)
        self.fc2 = nn.Linear(256, self.action_size)

    def forward(self, s):
        s = s.view(-1, self.in_channels, self.height, self.width) 

        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = s.view(-1, 64*(self.height//8)*(self.width//8))

        s = F.dropout(F.relu(self.fc1(s)), p=self.dropout, training=self.training)
        logits = self.fc2(s)

        return F.softmax(logits, 1)

class DQNet(nn.Module):
    def __init__(self, input_dim, action_size, dropout=0):
        self.in_channels, self.height, self.width= input_dim
        self.action_size = action_size
        self.dropout = dropout

        super(DQNet, self).__init__()
        self.conv1 = nn.Conv2d(self.in_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128*(self.height//8)*(self.width//8), 512)
        self.fc2 = nn.Linear(512, self.action_size)

    def forward(self, s):
        s = s.view(-1, self.in_channels, self.height, self.width) 

        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = s.view(-1, 128*(self.height//8)*(self.width//8))

        s = F.dropout(torch.sigmoid(self.fc1(s)), p=self.dropout, training=self.training)
        Q_values = self.fc2(s)

        return Q_values