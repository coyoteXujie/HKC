import torch
import Model

import torch.nn as nn

import numpy as np
import torch.optim as optim

from Model import MyDataset
from torch.utils.data import DataLoader


def trainModel(model, file_name):
    USE_CUDA = torch.cuda.is_available()

    save_name = ""
    dir_list = file_name.split("/")
    for i in range(len(dir_list) - 1):
        save_name += dir_list[i] + "/"

    boundary = len(np.load(save_name + "ClusterList.npy", allow_pickle=True))

    if USE_CUDA:
        net = model(boundary).cuda()
    else:
        net = model(boundary)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    trainset = MyDataset(save_name + "train_with_label.npy")
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0)

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if USE_CUDA:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    net.eval()
    torch.save(net.state_dict(), save_name + "net_latest.pth")


if __name__ == '__main__':
    ModleList = [Model.AudioQuery, Model.SunQuery, Model.EnronQuery,
                 Model.NuswideQuery, Model.NotreQuery]

    fileList = ["Zipf/audio/datasetKnn.hdf5", "Zipf/sun/datasetKnn.hdf5",
                "Zipf/enron/datasetKnn.hdf5", "Zipf/nuswide/datasetKnn.hdf5",
                "Zipf/notre/datasetKnn.hdf5"]


    for i in range(5):
        model = ModleList[i]
        file_name = fileList[i]
        print("train", file_name)
        trainModel(model, file_name)