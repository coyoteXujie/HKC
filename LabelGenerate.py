import random
import time

import utils as ut
import numpy as np

from tqdm import tqdm
from TreeStructure import Node, BinaryTree, ItemsTree


def ClusterTree(file_name, k=10, cluster_size=6):
    print(file_name)

    save_name = ""
    dir_list = file_name.split("/")
    for i in range(len(dir_list)-1):
        save_name += dir_list[i] + "/"

    # 登录数据集
    dataMatrix, trainDataset, trainKnn = ut.LoadDataset(file_name)

    data_num = dataMatrix.shape[0]

    # 构建聚类树基本信息
    pointsIndex = [i for i in range(data_num)]
    tree = BinaryTree()
    pointNum, cpoint = ut.CentralPointOfCluster(pointsIndex, dataMatrix)

    # 构造树的根节点
    tree.root = Node(cpoint)
    tree.root.pointnum = pointNum
    tree.root.Id = 0

    # 进行聚类
    start = time.perf_counter()
    tree.TDcluster(k, tree.root, pointsIndex, dataMatrix, cluster_size=cluster_size)
    end = time.perf_counter()
    timeUsable = end - start

    # 将构建的初始树的信息写入文件
    deep = tree.TreeInformation(tree.root)
    tree.postOrder(tree.root)
    with open(save_name + "树的相关信息.txt", "w") as file:
        file.write("{}数据集构建树完成，用时{}\n".format(file_name, timeUsable))
        file.write("\n" + "*" * 5 + "树的基本信息" + "*" * 5 + "\n")
        file.write("树的深度为：{}\n".format(deep))
        file.write("树的节点总数：{}\n".format(len(tree.nodes)))
        file.write("树的叶子节点数：{}\n".format(len(tree.leafNodes)))
        file.write("*" * 25 + "\n")

    print("聚类树完成，用时{}".format(timeUsable))

    # 保存树
    tree.SaveTree(save_name + "Tree.txt")
    print("树以保存至{}".format(save_name + "Tree.txt"))


def OptimizeIRepartionTree(file_name, k=10, cluster_size=6, portion_size=400):
    save_name = ""
    dir_list = file_name.split("/")
    for i in range(len(dir_list) - 1):
        save_name += dir_list[i] + "/"

    # 登录数据集
    dataMatrix, trainDataset, trainKnn = ut.LoadDataset(file_name)

    # 登录聚类树
    tree = ItemsTree()
    start = time.perf_counter()
    tree.LoadTree(save_name + "Tree.txt")
    end = time.perf_counter()
    with open(save_name + "树的相关信息.txt", "a") as f:
        f.write("\n针对Tree结构进行递增的重分配\n")
        f.write("登录Tree树结构用时：{}\n".format(end - start))
    print("\n针对Tree结构进行递增的重分配")
    print("登录Tree树结构用时：{}".format(end - start))

    start = time.perf_counter()
    tree.IncrementalRepartion(dataMatrix, file_name, k=k, cluster_size=cluster_size, portion_size=portion_size)
    end = time.perf_counter()
    with open(save_name + "树的相关信息.txt", "a") as f:
        f.write("优化时间为：{}\n".format(end - start))
    print("优化时间为：{}\n".format(end - start))

    tree.SaveTree(save_name + "IRepartionTree")


def LabelGeneration(file_name, c=1):
    save_name = ""
    dir_list = file_name.split("/")
    for i in range(len(dir_list) - 1):
        save_name += dir_list[i] + "/"

    # 登录聚类树
    tree = ItemsTree()
    start = time.perf_counter()
    tree.LoadTree(save_name + "IRepartionTree")
    end = time.perf_counter()
    with open(save_name + "树的相关信息.txt", "a") as f:
        f.write("\n生成神经网络的训练集\n")
        f.write("登录IRepartionTree结构用时：{}\n".format(end - start))
    print("\n生成神经网络的训练集")
    print("登录IRepartionTree结构用时：{}".format(end - start))


    # 这里需要清空是害怕 postOrder 曾被多次调用过
    tree.nodes = []
    tree.postOrder(tree.root)

    simple = Node([])
    nodeList = [simple for i in range(tree.recursion + 1)]

    for node in tree.nodes:
        if not nodeList[node.Id].cpoint:
            nodeList[node.Id] = node
        else:
            print("出现重复节点，代码有误")

    dataMatrix, trainDataset, trainKnn = ut.LoadDataset(file_name)

    label_dict = {}
    ClusterList = []
    flag = 0
    print(len(nodeList))
    for node in nodeList:

        if node.pointIndex is not None:
            ClusterList.append(node.pointIndex)
            label_dict[str(node.Id)] = flag
            flag += 1
    print(len(label_dict.keys()))
    print(len(ClusterList))
    np.save(save_name + "ClusterList", ClusterList)

    train_with_label = []
    for trainPoint in tqdm(trainDataset):

        sample = []
        candid, leafNodes = tree.search_C(trainPoint, c)
        index = label_dict[str(leafNodes[0].Id)]

        sample.append(trainPoint)
        sample.append(index)

        train_with_label.append(sample)

    np.save(save_name + "train_with_label", train_with_label)


def TestRepartionTree(file_name, c=1, k=10):
    save_name = ""
    dir_list = file_name.split("/")
    for i in range(len(dir_list) - 1) :
        save_name += dir_list[i] + "/"

    # 登录数据集
    dataMatrix, testDataset, testKnn = ut.LoadDataset(file_name)

    # 登录聚类树
    tree = ItemsTree()
    start = time.perf_counter()
    tree.LoadTree(save_name + "IRepartionTree")
    end = time.perf_counter()
    with open(save_name + "树的相关信息.txt", "a") as f:
        f.write("\n\n" + "#" * 5 + "测试IRepartionTree树的相关信息" + "#" * 5 + "\n")
        f.write("登录树结构用时：{}\n\n".format(end - start))
        print("\n" + "#" * 5 + "测试IRepartionTree树的相关信息" + "#" * 5)
        print("登录树结构用时：{}".format(end - start))

        start = time.perf_counter()
        for index in range(testDataset.shape[0]):
            testPoint = testDataset[index]
            candid, leafNodes = tree.search_C(testPoint, c)
            if len(candid) < k:
                predict, predictDistance = ut.Knn(len(candid), testPoint, candid, dataMatrix)
            else:
                predict, predictDistance = ut.Knn(k, testPoint, candid, dataMatrix)
        end = time.perf_counter()
        f.write("不添加重复元素的平均时间：{}\n".format((end - start) / testDataset.shape[0]))
        print("不添加重复元素的平均时间：{}".format((end - start) / testDataset.shape[0]))
        recall, ratio = ut.ShowResult(tree, dataMatrix, testDataset, testKnn,
                                      file_name.split('.')[0] + "Repartionc1nohave.txt", hava_element=False)
        f.write("不添加重复元素的召回率是：{}\n".format(recall))
        f.write("不添加重复元素的Ratio是：{}\n".format(ratio))


if __name__ == "__main__":
    # , "Zipf/sun/datasetKnn.hdf5",
    # "Zipf/enron/datasetKnn.hdf5", "Zipf/nuswide/datasetKnn.hdf5",
    # "Zipf/notre/datasetKnn.hd f5", "Zipf/sift/datasetKnn.hdf5"

    FileName = ["Zipf/audio/datasetKnn.hdf5"]
    cluster_size = 1
    for file_name in FileName:
        ClusterTree(file_name, k=10, cluster_size=cluster_size)
        OptimizeIRepartionTree(file_name, k=10, cluster_size=cluster_size, portion_size=400)
        # LabelGeneration(file_name, c=1)
        TestRepartionTree(file_name, c=1, k=10)