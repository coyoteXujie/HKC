

import numpy as np
import h5py as h5


def LoadDataset(file_name, test=False):

    with h5.File(file_name, "r") as f:
        dataMatrix = f["dataMatrix"]

        trainDataset = f["trainDataset"]
        trainKnn = f["trainKnn"]

        testDataset = f["testDataset"]
        testKnn = f["testKnn"]

        dataMatrix = np.asarray(dataMatrix)

        trainDataset = np.asarray(trainDataset)
        trainKnn = np.asarray(trainKnn)

        testDataset = np.asarray(testDataset)
        testKnn = np.asarray(testKnn)

        if test:
            result = (dataMatrix, testDataset, testKnn)

        else:
            result = (dataMatrix, trainDataset, trainKnn)

        return result


def Compute_Euclidean(v1, v2):
    """计算两点之间的欧式距离

    :param v1: 欧式空间中点1
    :param v2: 欧式空间中点2
    :return: 点1和点2之间的距离
    """
    sum_distance = 0
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    sum_distance += np.power(v1 - v2, 2).sum()
    distance = np.power(sum_distance, 0.5)

    return distance


def Compute_P(Tpointset_index, pointset_index):
    """ 计算桶的准确率

    :param Tpointset_index: 和某点相似的点的索引号
    :param pointset_index: 桶中分得点的索引号
    :return: 准确率
    """
    Tpointset_index = set(Tpointset_index)
    pointset_index = set(pointset_index)

    TP = len(Tpointset_index & pointset_index)

    TP_FP = len(pointset_index)

    if TP_FP == 0:
        precise = 0
        print("没有预测结果")
    else:
        precise = TP / TP_FP

    return precise


def Compute_R(Tpointset_index, pointset_index):
    """ 计算桶的准确率

    :param Tpointset_index: 和某点相似的点的索引号
    :param pointset_index: 桶中分得点的索引号
    :return: 召回率
    """
    Tpointset_index = set(Tpointset_index)
    pointset_index = set(pointset_index)

    TP = len(Tpointset_index & pointset_index)

    TP_FN = len(Tpointset_index)

    if TP_FN == 0:
        recall = 0
        print("没有正确结果")
    else:
        recall = TP / TP_FN

    return recall


def CentralPointOfCluster(pointsIndex, matrix):
    """ 计算聚类结果的中心点

    :param pointsIndex: 类中点的索引
    :param matrix: 类中点的各维度的数据
    :return: 聚类数量，聚类结果的中心点
    """
    matrix = np.asarray(matrix)

    cpoint = [0. for i in range(matrix.shape[1])]
    cpoint = np.asarray(cpoint)

    for pointIndex in pointsIndex:
        cpoint += np.asarray(matrix[pointIndex])

    pointNum = len(pointsIndex)
    # 这里有有个问题，当pointNum=0时，这里会有错误，但是之前都没有对pointsIndex的判空操作，这里留着吧
    if pointNum == 0:
        print("无中心点")
        return pointNum, cpoint
    else:
        cpoint = cpoint / pointNum
        cpoint = list(cpoint)

    return pointNum, cpoint


def Kmeans(cpoints, pointsIndex, matrix, iteration=15):
    """

    :param cpoints: 初始化的中心点
    :param pointsIndex: 需要聚类的点的Id的集合
    :param matrix: 点的真实位置
    :param iteration: 最大迭代次数
    :return: [中心点，中心点的聚类结果]
    """

    flag = 0  # 记录迭代的次数

    while True:

        # 保留原始的中心点
        originalCpoints = cpoints.copy()

        clusterPoints = [[] for i in range(len(cpoints))]  # 每一类包含的点的索引号
        for pointIndex in pointsIndex:
            minDistance = float("inf")
            selectCpoint = 0

            for cpointIndex in range(len(cpoints)):
                v1 = cpoints[cpointIndex]
                v2 = matrix[pointIndex]
                distance = Compute_Euclidean(v1, v2)
                if minDistance > distance:
                    minDistance = distance
                    selectCpoint = cpointIndex
            clusterPoints[selectCpoint].append(pointIndex)

        cpoints = []

        result = [[] for i in range(len(clusterPoints))]  # 最后的输出结果，0是中心点，1是中心点的聚类结果

        for i in range(len(clusterPoints)):
            cluster_pointsIndex = clusterPoints[i]

            if len(cluster_pointsIndex) == 0:
                continue

            _, cpoint = CentralPointOfCluster(cluster_pointsIndex, matrix)

            cpoints.append(cpoint)
            result[i].append(cpoint)
            result[i].append(cluster_pointsIndex)

        cpoints = np.asarray(cpoints)
        originalCpoints = np.asarray(originalCpoints)

        up_bound = cpoints < originalCpoints * 1.05
        low_bound = originalCpoints * 0.95 < cpoints

        if False not in up_bound:
            if False not in low_bound:
                print("聚类完成，共聚类{}次".format(flag))
                break

        flag += 1

        if flag > iteration:
            print("已经聚类{}次，时间过长，已经终止".format(iteration))
            break

    return result


def Knn(K, query, pointset_index, matrix):
    """ k近邻搜索方法

    :param k: 近邻数量
    :param query: 查询点
    :param pointset_index: 点集合的索引号
    :param matrix: 各个点的坐标
    :return: （query的K个最近邻在matrix中的索引号，query到K个最近邻的距离）
    """

    candid_matrix = [matrix[index] for index in pointset_index]
    candid_size = np.array(candid_matrix).shape[0]

    # 拼接为一个相同的矩阵
    different_matrix = np.tile(query, (candid_size, 1)) - candid_matrix

    # 做欧式距离的运算
    square_matrix = np.power(different_matrix, 2)
    sum_matrix = np.sum(square_matrix, axis=1)
    sqrt_matrix = np.power(sum_matrix, 0.5)

    K_indexes_temp = sqrt_matrix.argsort()[:K].tolist()

    K_distance = []
    for index in K_indexes_temp:
        K_distance.append(sqrt_matrix[index])

    K_indexes = []
    for data_index in K_indexes_temp:
        K_indexes.append(pointset_index[data_index])

    return K_indexes, K_distance


def ShowResult(tree, dataMatrix, testDataset, testKnn, file_name, k=10, c=1, hava_element=True):
    file = open(file_name, "w")

    meanPrecise = 0.
    meanRecall = 0.
    AVG = 0.

    for index in range(testDataset.shape[0]):
        flag_c = 2
        sumPDRD = 0.
        testPoint = testDataset[index]

        if hava_element:
            candid, leafNodes = tree.searchElement_C(testPoint, c)
        else:
            candid, leafNodes = tree.search_C(testPoint, c)

        while len(candid) == 0:
            candid, leafNodes = tree.search_C(testPoint, flag_c)
            flag_c += 1

        if len(candid) < k:
            predict, predictDistance = Knn(len(candid), testPoint, candid, dataMatrix)
        else:
            predict, predictDistance = Knn(k, testPoint, candid, dataMatrix)

        real, realDistance = testKnn[index]
        # print(predict)
        real = real[:len(predict)]
        # print(real)
        realDistance = realDistance[:len(predict)]

        precise = Compute_P(real, predict)
        recall = Compute_R(real, predict)
        leafNodesList = []
        for lnode in leafNodes:
            leafNodesList.append(lnode.Id)
        file.write("点的准确率是：{}，召回率是：{}\n".format(precise, recall))

        print("点的准确率是：{}，召回率是：{}".format(len(candid), recall))
        meanPrecise += precise
        meanRecall += recall
        for i in range(len(predict)):

            if predictDistance[i] == 0 or realDistance[i] == 0:
                sumPDRD += 1
                continue
            # print("预测的距离{}".format(predictDistance[i]))
            # print("真实的距离{}".format(realDistance[i]))
            # print(predictDistance[i] / realDistance[i])
            sumPDRD += predictDistance[i] / realDistance[i]
        sumPDRD = sumPDRD / len(predict)
        # print(sumPDRD)
        # print("*"*50)
        AVG += sumPDRD
    AVG = AVG / testDataset.shape[0]
    file.write("点的平均准确率是：{}\n".format(meanPrecise / testDataset.shape[0]))
    file.write("点的平均召回率是：{}\n".format(meanRecall / testDataset.shape[0]))
    file.write("AVG是：{}\n".format(AVG))

    print("点的平均准确率是：{}\n".format(meanPrecise / testDataset.shape[0]))
    print("点的平均召回率是：{}\n".format(meanRecall / testDataset.shape[0]))
    print("AVG是：{}\n".format(AVG))
    file.close()

    return meanRecall / testDataset.shape[0], AVG


if __name__ == "__main__":

    LoadDataset("Zipf/audio/datasetKnn.hdf5")


    # # 测试Compute_Euclidean
    # v1 = [1, 2, 3]
    # v2 = [2, 3, 4]
    # a = Compute_Euclidean(v1, v2)
    # print(a)

    # # 测试Kmeans
    # matrix = [[1, 2, 3],
    #           [4, 5, 6],
    #           [7, 8, 9],
    #           [5, 5, 5]]
    #
    # pointsIndex = [0, 1, 2, 3]
    #
    # cpoints = [[1, 1, 1],
    #            [9, 9, 9]]
    # result = Kmeans(cpoints, pointsIndex, matrix)
    # for node in result:
    #     print(node)


    # 测试Knn
    matrix = [[1, 0, 0, 0, 0],
              [1, 0, 0, 0, 0],
              [1, 2, 3, 3, 3],
              [1, 2, 2, 2, 2],
              [1, 2, 2, 3, 3]]
    pointset_index = [0, 1, 2, 3, 4]
    point = [1, 1, 1, 1, 1]
    K_list, _ = Knn(4, point, pointset_index, matrix)
    print(K_list)

    pass