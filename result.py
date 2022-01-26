import torch
import Model
import time

import numpy as np
import utils as ut

from torch.utils.data import Dataset, DataLoader


USE_CUDA = torch.cuda.is_available()

class TestDataset(Dataset):
    def __init__(self, file_name):
        super(TestDataset, self).__init__()
        dataMatrix, testDataset, testKnn = ut.LoadDataset(file_name, test=True)

        self.testDataset = testDataset
        self.testKnn = testKnn

    def __len__(self):

        return len(self.testDataset)

    def __getitem__(self, idx):

        query = torch.Tensor([self.testDataset[idx]])
        Knn = self.testKnn[idx]
        return query, Knn


def test_model_c(model, file_name):
    save_name = ""
    dir_list = file_name.split("/")
    for i in range(len(dir_list) - 1):
        save_name += dir_list[i] + "/"

    dataMatrix, testDataset, testKnn = ut.LoadDataset(file_name, test=True)
    ClusterList = np.load(save_name + "ClusterList.npy", allow_pickle=True)
    boundary = len(ClusterList)

    testset = TestDataset(file_name)
    trainloader = DataLoader(testset, batch_size=128, shuffle=True, num_workers=0)

    net = model(boundary)
    net.load_state_dict(torch.load(save_name + "net_latest.pth"))
    net.eval()

    SumTime = 0.
    meanPrecise = 0.
    AVG = 0.
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):

            inputs, Knns = data
            break1 = time.perf_counter()
            outputs = net(inputs)
            break2 = time.perf_counter()
            SumTime += break2 - break1

            for index in range(len(inputs)):
                sumPDRD = 0
                query = inputs[index, 0].tolist()
                real, realDistance = Knns[index]
                result = outputs[index]

                candid = []

                _, predicted = torch.topk(result.data, 1, 0)

                for cluster in predicted:
                    candid.extend(ClusterList[cluster])

                break3 = time.perf_counter()
                K_list, K_distance = ut.Knn(10, query, candid, dataMatrix)
                break4 = time.perf_counter()
                SumTime += break4 - break3

                real = real[:len(K_list)]
                realDistance = realDistance[:len(K_list)]
                real = real.tolist()
                realDistance = realDistance.tolist()

                precise = ut.Compute_P(real, K_list)
                # print("点的准确率是：{}，召回率是：{}".format(precise, precise))

                meanPrecise += precise

                for i in range(len(K_list)):
                    if K_distance[i] == 0 or realDistance[i] == 0:
                        sumPDRD += 1
                        continue

                    sumPDRD += K_distance[i] / realDistance[i]

                sumPDRD = sumPDRD / len(K_list)
                AVG += sumPDRD

    print("查询点的平均时间是：{}\n".format(SumTime / testDataset.shape[0]))
    print("点的平均准确率是：{}\n".format(meanPrecise / testDataset.shape[0]))
    print("AVG是：{}\n".format(AVG / testDataset.shape[0]))

    with open("result.txt", "a") as f:
        f.write("\n\n数据集 {} 的结果\n".format(dir_list[1]))
        f.write("查询点的平均时间是：{}\n".format(SumTime / testDataset.shape[0]))
        f.write("点的平均准确率是：{}\n".format(meanPrecise / testDataset.shape[0]))
        f.write("AVG是：{}\n".format(AVG / testDataset.shape[0]))


def test_model_g(model, file_name):
    save_name = ""
    dir_list = file_name.split("/")
    for i in range(len(dir_list) - 1):
        save_name += dir_list[i] + "/"

    dataMatrix, testDataset, testKnn = ut.LoadDataset(file_name, test=True)
    ClusterList = np.load(save_name + "ClusterList.npy", allow_pickle=True)
    boundary = len(ClusterList)

    testset = TestDataset(file_name)
    trainloader = DataLoader(testset, batch_size=128, shuffle=True, num_workers=0)

    if USE_CUDA:
        net = model(boundary).cuda()
    else:
        net = model(boundary)

    net.load_state_dict(torch.load(save_name + "net_latest.pth"))
    net.eval()

    SumTime = 0.
    meanPrecise = 0.
    AVG = 0.
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):

            inputs, Knns = data
            break1 = time.perf_counter()
            outputs = net(inputs)
            break2 = time.perf_counter()
            SumTime += break2 - break1

            for index in range(len(inputs)):
                sumPDRD = 0
                query = inputs[index, 0].tolist()
                real, realDistance = Knns[index]
                result = outputs[index]

                candid = []

                _, predicted = torch.topk(result.data, 1, 0)

                for cluster in predicted:
                    candid.extend(ClusterList[cluster])

                break3 = time.perf_counter()
                K_list, K_distance = ut.Knn(10, query, candid, dataMatrix)
                break4 = time.perf_counter()
                SumTime += break4 - break3

                real = real[:len(K_list)]
                realDistance = realDistance[:len(K_list)]
                real = real.tolist()
                realDistance = realDistance.tolist()

                precise = ut.Compute_P(real, K_list)
                # print("点的准确率是：{}，召回率是：{}".format(precise, precise))

                meanPrecise += precise

                for i in range(len(K_list)):
                    if K_distance[i] == 0 or realDistance[i] == 0:
                        sumPDRD += 1
                        continue

                    sumPDRD += K_distance[i] / realDistance[i]

                sumPDRD = sumPDRD / len(K_list)
                AVG += sumPDRD

    print("查询点的平均时间是：{}\n".format(SumTime / testDataset.shape[0]))
    print("点的平均准确率是：{}\n".format(meanPrecise / testDataset.shape[0]))
    print("AVG是：{}\n".format(AVG / testDataset.shape[0]))



if __name__ == '__main__':
    ModleList = [Model.AudioQuery]

    fileList = ["Zipf/audio/datasetKnn.hdf5"]


    for i in range(5):
        model = ModleList[i]
        file_name = fileList[i]
        test_model_c(model, file_name)
