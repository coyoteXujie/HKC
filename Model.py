# encoding: utf-8
import torch

import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, file_name):
        super(MyDataset, self).__init__()

        self.trainDataset = np.load(file_name, allow_pickle=True)

    def __len__(self):

        return len(self.trainDataset)

    def __getitem__(self, idx):

        query = torch.Tensor([self.trainDataset[idx, 0]])
        target = self.trainDataset[idx, 1]

        return query, target


class AudioQuery(nn.Module):
    def __init__(self, out_dim):
        super(AudioQuery, self).__init__()

        self.out_dim = out_dim

        # 第一层，3个卷积层和一个最大池化层
        self.layer1 = nn.Sequential(
            # 输入1通道，卷积核长度为3，输出32通道（如192的向量，(192+2*1-3)/2+1=96，输出96*32）
            nn.Conv1d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # 输入32通道，卷积核3，输出32通道（输入96*32，卷积3*32*32，输出96*32）
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # 输入32通道，卷积核3，输出32通道（输入96*32，卷积3*32*32，输出96*32）
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # 输入96*32，输出48*32
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第二层，2个卷积层和一个最大池化层
        self.layer2 = nn.Sequential(
            # 输入32通道，卷积核长度为3，输出64通道（输入48*32，卷积3*32*64，输出24*64）
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # 输入64通道，卷积核3，输出64通道（输入24*64，卷积3*64*64，输出24*64）
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # 输入24*32，输出12*32
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第三层，2个卷积层和一个最大池化层
        self.layer3 = nn.Sequential(
            # 输入64通道，卷积核3，输出128通道（输入12*64，卷积3*64*128，输出6*128）
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # 输入128通道，卷积核3，输出128通道（输入6*128，卷积3*128*128，输出6*128）
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # 输入6*128，输出3*128
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv_layer = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3
        )

        self.fc = nn.Sequential(
            nn.Linear(3 * 128, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(1024, self.out_dim)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 3 * 128)
        x = self.fc(x)

        return x


class SunQuery(nn.Module):
    def __init__(self, out_dim):
        super(SunQuery, self).__init__()

        self.out_dim = out_dim

        # 第一层，3个卷积层和一个最大池化层
        self.layer1 = nn.Sequential(
            # 输入1通道，卷积核长度为3，输出32通道（如512的向量，(512+2*1-3)/2+1=256，输出256*32）
            nn.Conv1d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # 输入32通道，卷积核3，输出32通道（输入256*32，卷积3*32*32，输出256*32）
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # 输入32通道，卷积核3，输出32通道（输入256*32，卷积3*32*32，输出256*32）
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # 输入256*32，输出128*32
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第二层，2个卷积层和一个最大池化层
        self.layer2 = nn.Sequential(
            # 输入32通道，卷积核长度为3，输出64通道（输入128*32，卷积3*32*64，输出64*64）
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # 输入64通道，卷积核3，输出64通道（输入64*64，卷积3*64*64，输出64*64）
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # 输入64*64，输出32*64
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第三层，3个卷积层和一个最大池化层
        self.layer3 = nn.Sequential(
            # 输入64通道，卷积核3，输出128通道（输入32*64，卷积3*64*128，输出16*128）
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # 输入128通道，卷积核3，输出128通道（输入16*128，卷积3*128*128，输出16*128）
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # 输入128通道，卷积核3，输出128通道（输入16*128，卷积3*128*128，输出16*128）
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # 输入16*128，输出8*128
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv_layer = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * 128, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(1024, self.out_dim)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 8 * 128)
        x = self.fc(x)

        return x


class EnronQuery(nn.Module):
    def __init__(self, out_dim):
        super(EnronQuery, self).__init__()
        self.out_dim = out_dim

        # 第一层，3个卷积层和一个最大池化层
        self.layer1 = nn.Sequential(
            # 输入1通道，卷积核长度为3，输出32通道（如1369的向量，(1369+2*1-3)/2+1=685，输出685*32）
            nn.Conv1d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # 输入32通道，卷积核3，输出32通道（输入685*32，卷积3*32*32，输出685*32）
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # 输入32通道，卷积核3，输出32通道（输入685*32，卷积3*32*32，输出685*32）
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # 输入685*32，输出342*32
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第二层，2个卷积层和一个最大池化层
        self.layer2 = nn.Sequential(
            # 输入32通道，卷积核长度为3，输出64通道（输入342*32，卷积3*32*64，输出171*64）
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # 输入64通道，卷积核3，输出64通道（输入171*64，卷积3*64*64，输出171*64）
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # 输入171*64，输出85*64
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第三层，3个卷积层和一个最大池化层
        self.layer3 = nn.Sequential(
            # 输入64通道，卷积核3，输出128通道（输入85*64，卷积3*64*128，输出43*128）
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # 输入128通道，卷积核3，输出128通道（输入43*128，卷积3*128*128，输出43*128）
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # 输入128通道，卷积核3，输出128通道（输入43*128，卷积3*128*128，输出43*128）
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # 输入43*128，输出21*128
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第四层，3个卷积层和一个最大池化层
        self.layer4 = nn.Sequential(
            # 输入128通道，卷积核3，输出256通道（输入21*128，卷积3*128*256，输出11*256）
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # 输入256通道，卷积核3，输出256通道（输入11*256，卷积3*256*256，输出11*256）
            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # 输入256通道，卷积核3，输出256通道（输入11*256，卷积3*256*256，输出11*256）
            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # 输入11*256，输出5*256
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv_layer = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        )

        self.fc = nn.Sequential(
            nn.Linear(5 * 256, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(1024, self.out_dim)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 5 * 256)
        x = self.fc(x)

        return x


class NuswideQuery(nn.Module):
    def __init__(self, out_dim):
        super(NuswideQuery, self).__init__()

        self.out_dim = out_dim

        # 第一层，3个卷积层和一个最大池化层
        self.layer1 = nn.Sequential(
            # 输入1通道，卷积核长度为3，输出32通道（如500的向量，(500+2*1-3)/2+1=250，输出250*32）
            nn.Conv1d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # 输入32通道，卷积核3，输出32通道（输入250*32，卷积3*32*32，输出250*32）
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # 输入32通道，卷积核3，输出32通道（输入250*32，卷积3*32*32，输出250*32）
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # 输入250*32，输出125*32
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第二层，2个卷积层和一个最大池化层
        self.layer2 = nn.Sequential(
            # 输入32通道，卷积核长度为3，输出64通道（输入125*32，卷积3*32*64，输出63*64）
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # 输入64通道，卷积核3，输出64通道（输入63*64，卷积3*64*64，输出63*64）
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # 输入63*64，输出31*64
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第三层，3个卷积层和一个最大池化层
        self.layer3 = nn.Sequential(
            # 输入64通道，卷积核3，输出128通道（输入31*64，卷积3*64*128，输出16*128）
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # 输入128通道，卷积核3，输出128通道（输入16*128，卷积3*128*128，输出16*128）
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # 输入128通道，卷积核3，输出128通道（输入16*128，卷积3*128*128，输出16*128）
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # 输入16*128，输出8*128
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv_layer = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * 128, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(1024, self.out_dim)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 8 * 128)
        x = self.fc(x)

        return x


class NotreQuery(nn.Module):

    def __init__(self, out_dim):
        super(NotreQuery, self).__init__()

        self.out_dim = out_dim

        # 第一层，2个卷积层和一个最大池化层
        self.layer1 = nn.Sequential(
            # 输入1通道，卷积核长度为3，输出32通道（如128的向量，(128+2*1-3)/2+1=64，输出64*32）
            nn.Conv1d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # 输入32通道，卷积核3，输出32通道（输入64*32，卷积3*32*32，输出64*32）
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # 输入32通道，卷积核3，输出32通道（输入64*32，卷积3*32*32，输出64*32）
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # 输入64*32，输出32*32
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第二层，2个卷积层和一个最大池化层
        self.layer2 = nn.Sequential(
            # 输入32通道，卷积核长度为3，输出65通道（输入32*32，卷积3*32*64，输出16*64）
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # 输入64通道，卷积核3，输出64通道（输入16*64，卷积3*64*64，输出16*64）
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # 输入16*64，输出8*64
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第三层，3个卷积层和一个最大池化层
        self.layer3 = nn.Sequential(
            # 输入64通道，卷积核3，输出128通道（输入8*64，卷积3*64*128，输出4*128）
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # 输入128通道，卷积核3，输出128通道（输入4*128，卷积3*128*128，输出4*128）
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # 输入128通道，卷积核3，输出128通道（输入4*128，卷积3*128*128，输出4*128）
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # 输入4*128，输出2*128
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv_layer = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * 128, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(1024, self.out_dim)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 2 * 128)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    x = torch.Tensor([i for i in range(128)])
    x = x.unsqueeze(0)
    x = x.unsqueeze(0)

    net = NotreQuery(2000)
    x = net(x)
    print(x.shape)