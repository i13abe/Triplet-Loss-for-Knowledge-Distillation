import torch
import torch.nn as nn

class Net_teacher(nn.Module):
    def __init__(self):
        super(Net_teacher, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3, padding=1)
        self.conv2 = nn.Conv2d(32,32,3, padding=1)
        self.conv3 = nn.Conv2d(32,64, 3, padding=1)
        self.conv4 = nn.Conv2d(64,64, 3, padding=1)
        self.conv5 = nn.Conv2d(64,128, 3, padding=1)
        self.fc1 = nn.Linear(128*4*4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.batchnorm5 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.relu(self.conv4(x))
        x = self.batchnorm4(x)
        x = self.relu(self.conv5(x))
        x = self.batchnorm5(x)
        x = self.pool(x)

        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class Net_student(nn.Module):
    def __init__(self):
        super(Net_student, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(-1, 64 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x