import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 22), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 11))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(512, 4)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0,1,3,2)
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.3)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.3)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.3)
        x = self.pooling3(x)

        # FC Layer
        x = x.reshape(-1,512)
        x = self.fc1(x)
        return x

class EEGNet_2b(nn.Module):
    def __init__(self):
        super(EEGNet_2b, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 3), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 11))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(536, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0,1,3,2)
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.3)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.3)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.3)
        x = self.pooling3(x)

        # FC Layer
        x = x.reshape(-1,536)
        x = self.fc1(x)
        return x




class EEGNet_500interval(nn.Module):
    def __init__(self):
        super(EEGNet_500interval, self).__init__()
        self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 22), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 11))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # Layer 4
        self.padding3 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv4 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm4 = nn.BatchNorm2d(4, False)
        self.pooling4 = nn.MaxPool2d((2, 2))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32,4)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0,1,3,2)
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.3)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.3)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.3)
        x = self.pooling3(x)

        # Layer 4
        x = self.padding3(x)
        x = F.elu(self.conv4(x))
        x = self.batchnorm4(x)
        x = F.dropout(x, 0.3)
        x = self.pooling4(x)

        # FC Layer
        x = x.reshape(-1,64)
        x = F.elu(self.fc1(x))

        # FC Layer2
        x = F.elu(self.fc2(x))

        return x


class DeepConvNet(nn.Module):
    def __init__(self, channels = 22, dropout = 0.2, nb_class = 4):
        super(DeepConvNet, self).__init__()
        self.dropout = dropout
        self.nb_class = nb_class
        # Layer 1
        self.conv1 = nn.Conv2d(1, 25, (1, 5), padding=0)
        self.conv1_2 =nn.Conv2d(25,25,(channels,1))
        self.batchnorm1 = nn.BatchNorm2d(25, False)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(1,2) ,stride=(1,2))

        # Layer 2
        self.conv2 = nn.Conv2d(25, 50, (1, 5))
        self.batchnorm2 = nn.BatchNorm2d(50, False)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(1,2) ,stride=(1,2))

        # Layer 3
        self.conv3 = nn.Conv2d(50, 100, (1, 5))
        self.batchnorm3 = nn.BatchNorm2d(100, False)
        self.maxpooling3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Layer 4
        self.conv4 = nn.Conv2d(100, 200, (1, 5))
        self.batchnorm4 = nn.BatchNorm2d(200, False)
        self.maxpooling4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(100*121, self.nb_class)

    def forward(self, x):
        x = x.unsqueeze(1)
        # Layer 1
        x = self.conv1(x)
        x = self.conv1_2(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        x = self.maxpooling1(x)
        x = F.dropout(x, self.dropout)
        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.maxpooling2(x)
        x = F.dropout(x, self.dropout)

        # Layer 3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.maxpooling3(x)
        x = F.dropout(x, self.dropout)

        # Flatten
        x = x.reshape(-1,100*121)
        # FC Layer
        x = self.fc1(x)
        #
        # # FC Layer2
        # x = F.elu(self.fc2(x))

        return x

class DeepConvNet_2b(nn.Module):
    def __init__(self, channels = 22, dropout = 0.2, nb_class = 4):
        super(DeepConvNet_2b, self).__init__()
        self.dropout = dropout
        self.nb_class = nb_class
        # Layer 1
        self.conv1 = nn.Conv2d(1, 25, (1, 5), padding=0)
        self.conv1_2 =nn.Conv2d(25,25,(channels,1))
        self.batchnorm1 = nn.BatchNorm2d(25, False)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(1,2) ,stride=(1,2))

        # Layer 2
        self.conv2 = nn.Conv2d(25, 50, (1, 5))
        self.batchnorm2 = nn.BatchNorm2d(50, False)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(1,2) ,stride=(1,2))

        # Layer 3
        self.conv3 = nn.Conv2d(50, 100, (1, 5))
        self.batchnorm3 = nn.BatchNorm2d(100, False)
        self.maxpooling3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Layer 4
        self.conv4 = nn.Conv2d(100, 200, (1, 5))
        self.batchnorm4 = nn.BatchNorm2d(200, False)
        self.maxpooling4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(100*127, self.nb_class)

    def forward(self, x):
        x = x.unsqueeze(1)
        # Layer 1
        x = self.conv1(x)
        x = self.conv1_2(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        x = self.maxpooling1(x)
        x = F.dropout(x, self.dropout)
        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.maxpooling2(x)
        x = F.dropout(x, self.dropout)

        # Layer 3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.maxpooling3(x)
        x = F.dropout(x, self.dropout)

        # Flatten
        x = x.reshape(-1,100*127)
        # FC Layer
        x = self.fc1(x)
        #
        # # FC Layer2
        # x = F.elu(self.fc2(x))

        return x


# class Shallow_Net(nn.Module):
#     def __init__(self, channels = 22, dropout = 0.2, nb_class = 4):
#         super(Shallow_Net, self).__init__()
#         self.dropout = dropout
#         self.nb_class = nb_class
#         # Layer 1
#         self.conv1 = nn.Conv2d(1, 40, (1, 13), padding=0)
#
#         # Layer 2
#         self.conv2 = nn.Conv2d(40, 40, (channels, 1),bias=False)
#         self.batchnorm = nn.BatchNorm2d(40)
#         # self.pooling2 = nn.AvgPool2d(kernel_size=(1,35), stride=(1,7))
#         self.pooling2 = nn.AvgPool2d(kernel_size=(1, 30), stride=(1, 4))
#
#         # FC Layer
#         # NOTE: This dimension will depend on the number of timestamps per sample in your data.
#         # I have 120 timepoints.
#         # self.fc1 = nn.Linear(137*40, self.nb_class)
#         self.fc1 = nn.Linear(240 * 40, self.nb_class)
#
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         # Layer 1
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.batchnorm(x)
#         x = F.relu(x)
#         x = self.pooling2(x)
#         x = F.dropout(x, self.dropout)
#         x = x.squeeze(2)
#         x = x.reshape(-1,x.shape[1]*x.shape[2])
#         # FC Layer
#         x = self.fc1(x)
#
#         return x


class Shallow_Net(nn.Module):
    def __init__(self, channels = 22, dropout = 0.2, nb_class = 4):
        super(Shallow_Net, self).__init__()
        self.dropout = dropout
        self.nb_class = nb_class
        # Layer 1
        self.conv1 = nn.Conv2d(1, 40, (1, 13), padding=0)

        # Layer 2
        self.conv2 = nn.Conv2d(40, 40, (channels, 1),bias=False)
        self.batchnorm = nn.BatchNorm2d(40)
        # self.pooling2 = nn.AvgPool2d(kernel_size=(1,35), stride=(1,7))
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 30), stride=(1, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        # self.fc1 = nn.Linear(137*40, self.nb_class)
        self.fc1 = nn.Linear(240 * 40, self.nb_class)

    def forward(self, x):
        x = x.unsqueeze(1)
        # Layer 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.pooling2(x)
        x = F.dropout(x, self.dropout)
        x = x.squeeze(2)
        x = x.reshape(-1,x.shape[1]*x.shape[2])
        # FC Layer
        x = self.fc1(x)

        return x

class Shallow_Net_2b(nn.Module):
    def __init__(self, channels = 22, dropout = 0.2, nb_class = 2):
        super(Shallow_Net_2b, self).__init__()
        self.dropout = dropout
        self.nb_class = nb_class
        # Layer 1
        self.conv1 = nn.Conv2d(1, 40, (1, 15), padding=0)

        # Layer 2
        self.conv2 = nn.Conv2d(40, 40, (channels, 1),bias=False)
        self.batchnorm = nn.BatchNorm2d(40)
        # self.pooling2 = nn.AvgPool2d(kernel_size=(1,35), stride=(1,7))
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 20), stride=(1, 2))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        # self.fc1 = nn.Linear(137*40, self.nb_class)
        self.fc1 = nn.Linear(20360, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        # Layer 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.pooling2(x)
        x = F.dropout(x, self.dropout)
        x = x.squeeze(2)
        x = x.reshape(-1,x.shape[1]*x.shape[2])
        # FC Layer
        x = self.fc1(x)

        return x


if __name__ == "__main__":
    inp = torch.autograd.Variable(torch.randn(2, 22, 500))
    model = DeepConvNet(22,0.2,4)
    output = model(inp)


