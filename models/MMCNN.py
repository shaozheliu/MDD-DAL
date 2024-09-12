import math
import torch
import collections
import numpy as np
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple


#
# import torch
# module = torch.nn.Conv1d(20, 20, 5, padding='same')
# x = torch.randn(1, 20, 20)
# w = module(x)
# print(w.shape)




#### inception_block ####
class inception_block(nn.Module):
    def __init__(self,channels, ince_filter, ince_length):
        super(inception_block, self).__init__()
        # Layer 1
        self.k1, self.k2, self.k3, self.k4 = ince_filter
        self.l1, self.l2, self.l3, self.l4 = ince_length
        self.conv1 = nn.Conv1d(channels, out_channels = self.k1, kernel_size=self.l1, padding='same')
        self.batchnorm1 = nn.BatchNorm1d(self.k1, False)
        # Layer 2
        self.conv2 = nn.Conv1d(channels, out_channels=self.k2, kernel_size=self.l2, padding='same')
        self.batchnorm2 = nn.BatchNorm1d(self.k2, False)
        # Layer 3
        self.conv3 = nn.Conv1d(channels, out_channels=self.k3, kernel_size=self.l1, padding='same')
        self.batchnorm3 = nn.BatchNorm1d(self.k3, False)
        # Layer 4
        self.conv4 = nn.Conv1d(channels, out_channels=self.k4, kernel_size=self.l1, padding='same')
        self.batchnorm4 = nn.BatchNorm1d(self.k4, False)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.batchnorm1(x1)
        x1 = F.elu(x1)
        x2 = self.conv2(x)
        x2 = self.batchnorm2(x2)
        x2 = F.elu(x2)
        x3 = self.conv3(x)
        x3 = self.batchnorm3(x3)
        x3 = F.elu(x3)
        x4 = self.conv4(x)
        x4 = self.batchnorm4(x4)
        x4 = F.elu(x4)
        res = torch.cat([x1,x2,x3,x4],axis = 1)
        return res

#### conv block
class conv_block(nn.Module):
    def __init__(self,channels, nb_filter, length, dropout):
        super(conv_block, self).__init__()
        # Layer 1
        self.k1, self.k2, self.k3 = nb_filter
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels = self.k1, kernel_size=length, padding='same')
        self.batchnorm1 = nn.BatchNorm1d(self.k1, False)
        # Layer 2
        self.conv2 = nn.Conv1d(self.k1, out_channels=self.k2, kernel_size=length, padding='same')
        self.batchnorm2 = nn.BatchNorm1d(self.k2, False)
        # Layer 3
        self.conv3 = nn.Conv1d(self.k2, out_channels=self.k3, kernel_size=length, padding='same')
        self.batchnorm3 = nn.BatchNorm1d(self.k3, False)
        # Layer 4
        self.conv4 = nn.Conv1d(self.k3, out_channels=1, kernel_size=1, padding='same')
        self.batchnorm4 = nn.BatchNorm1d(1, False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = F.elu(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = F.elu(out)
        out = self.conv3(out)
        out = self.batchnorm3(out)
        out = F.elu(out)
        out = self.conv4(out)
        out = self.batchnorm4(out)
        out = F.elu(out + x)
        out = self.dropout(out)

        return out


#### se_block
class squeeze_excitation_layer(nn.Module):
    def __init__(self, out_dim,activation,ratio=8):
        super(squeeze_excitation_layer, self).__init__()
        self.linear1 = nn.Linear(5,out_dim//ratio)
        # self.linear2 = nn.Linear(out_dim//ratio, out_dim)

    def forward(self, x):
        x = x.mean(dim = 1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class MMCNN_model(nn.Module):
    def __init__(self, channels = 3, samples = 1000, dropoutrat = 0.4):
        super(MMCNN_model, self).__init__()
        self.channels = channels
        self.samples = samples
        self.dropout = nn.Dropout(dropoutrat)
        # the parameter of the first part :EEG Inception block
        self.inception_filters = [16, 16, 16, 16]
        self.inception_kernel_length = [[5, 10, 15, 10],
                                        [40, 45, 50, 100],
                                        [60, 65, 70, 100],
                                        [80, 85, 90, 100],
                                        [160, 180, 200, 180], ]
        self.inception_stride = [2, 4, 4, 4, 16]
        self.first_maxpooling_size = 4
        self.first_maxpooling_stride = 4
        # the parameter of the second part :Residual block
        self.res_block_filters = [16, 16, 16]
        self.res_block_kernel_stride = [8, 7, 7, 7, 6]
        # the parameter of the third part :SE block
        self.se_block_kernel_stride = 16
        self.se_ratio = 8
        self.second_maxpooling_size = [4, 3, 3, 3, 2]
        self.second_maxpooling_stride = [4, 3, 3, 3, 2]

        # inception block1
        self.inception_block1 = inception_block(self.samples, self.inception_filters, self.inception_kernel_length[0])
        self.maxpooling1 = nn.MaxPool1d(kernel_size=self.first_maxpooling_size, stride=self.first_maxpooling_stride)
        self.batchnorm1 = nn.BatchNorm1d(sum(self.inception_filters))

        # conv block1
        self.conv_block1 = conv_block(sum(self.inception_filters), self.res_block_filters, self.res_block_kernel_stride[0], dropoutrat)

        # inception block2
        self.inception_block2 = inception_block(self.samples, self.inception_filters, self.inception_kernel_length[1])
        self.maxpooling2 = nn.MaxPool1d(kernel_size=self.first_maxpooling_size, stride=self.first_maxpooling_stride)
        self.batchnorm2 = nn.BatchNorm1d(sum(self.inception_filters))

        # conv block2
        self.conv_block2 = conv_block(sum(self.inception_filters), self.res_block_filters,
                                      self.res_block_kernel_stride[1], dropoutrat)

        # inception block3
        self.inception_block3 = inception_block(self.samples, self.inception_filters, self.inception_kernel_length[2])
        self.maxpooling3 = nn.MaxPool1d(kernel_size=self.first_maxpooling_size, stride=self.first_maxpooling_stride)
        self.batchnorm3 = nn.BatchNorm1d(sum(self.inception_filters))

        # conv block3
        self.conv_block3 = conv_block(sum(self.inception_filters), self.res_block_filters,
                                      self.res_block_kernel_stride[2], dropoutrat)

        # inception block4
        self.inception_block4 = inception_block(self.samples, self.inception_filters, self.inception_kernel_length[3])
        self.maxpooling4 = nn.MaxPool1d(kernel_size=self.first_maxpooling_size, stride=self.first_maxpooling_stride)
        self.batchnorm4 = nn.BatchNorm1d(sum(self.inception_filters))

        # conv block4
        self.conv_block4 = conv_block(sum(self.inception_filters), self.res_block_filters,
                                      self.res_block_kernel_stride[3], dropoutrat)

        # inception block5
        self.inception_block5 = inception_block(self.samples, self.inception_filters, self.inception_kernel_length[4])
        self.maxpooling5 = nn.MaxPool1d(kernel_size=self.first_maxpooling_size, stride=self.first_maxpooling_stride)
        self.batchnorm5 = nn.BatchNorm1d(sum(self.inception_filters))

        # conv block5
        self.conv_block5 = conv_block(sum(self.inception_filters), self.res_block_filters,
                                      self.res_block_kernel_stride[4], dropoutrat)

        # squeez ex block1
        self.squeeze_excitation_layer = squeeze_excitation_layer(self.se_block_kernel_stride, self.se_ratio)
        self.linear = nn.Linear(15,4)
    def forward(self, x):
        x = x.permute(0,2,1)

        ## EIN-a
        # inception
        x1 = self.inception_block1(x)  # 时间维度上压缩  batch,64,22
        x1 = self.maxpooling1(x1)
        x1 = self.batchnorm1(x1)
        x1 = self.dropout(x1)
        # conv
        x1 = self.conv_block1(x1)
        # x1 = self.squeeze_ex1citation_layer(x1)

        ## EIN-b
        # inception
        x2 = self.inception_block2(x)  # 时间维度上压缩  batch,64,22
        x2 = self.maxpooling2(x2)
        x2 = self.batchnorm2(x2)
        x2 = self.dropout(x2)
        # conv
        x2 = self.conv_block2(x2)
        # x2 = self.squeeze_ex2citation_layer(x2)

        ## EIN-c
        # inception
        x3 = self.inception_block3(x)  # 时间维度上压缩  batch,64,33
        x3 = self.maxpooling3(x3)
        x3 = self.batchnorm3(x3)
        x3 = self.dropout(x3)
        # conv
        x3 = self.conv_block3(x3)
        # x3 = self.squeeze_ex3citation_layer(x3)

        ## EIN-d
        # inception
        # x4 = self.inception_block4(x)  # 时间维度上压缩  batch,64,44
        # x4 = self.maxpooling4(x4)
        # x4 = self.batchnorm4(x4)
        # x4 = self.dropout(x4)
        # # conv
        # x4 = self.conv_block4(x4)
        # # x4 = self.squeeze_ex4citation_layer(x4)
        #
        # ## EIN-e
        # # inception
        # x5 = self.inception_block5(x)  # 时间维度上压缩  batch,65,55
        # x5 = self.maxpooling5(x5)
        # x5 = self.batchnorm5(x5)
        # x5 = self.dropout(x5)
        # # conv
        # x5 = self.conv_block5(x5)
        # x5 = self.squeeze_ex5citation_layer(x5)
        x1 = x1.mean(axis=1)
        x2 = x2.mean(axis=1)
        x3 = x3.mean(axis=1)
        # x4 = x4.mean(axis=1)
        # x5 = x5.mean(axis=1)
        res = torch.cat([x1,x2,x3],axis=-1)
        res = self.linear(res)
        return res





if __name__ == '__main__':
    model = MMCNN_model(22,1000)
    input = torch.randn(250, 22, 1000)
    out = model(input)





