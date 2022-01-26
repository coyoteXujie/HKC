

import time
import random
import sys
import munkres

import numpy as np
import utils as ut

from tqdm import tqdm
from munkres import Munkres, DISALLOWED



class Node:
    def __init__(self, cpoint, pointIndex=None):
        """二叉树的节点

        :param cpoint: 该节点的中心点
        :param pointIndex: 叶子节点所代表的集合 ，非叶子节点这里是None
        """

        # 节点的基本信息
        self.cpoint = cpoint
        self.pointIndex = pointIndex
        self.pointnum = None  # 用来标记当前节点的子树一共包含多少个点

        self.Id = None

        self.father = None  # 用来标记当前节点的父节点
        self.lchild = None  # 左孩子
        self.rchild = None  # 右孩子

        self.fatherId = None
        self.lchildId = None
        self.rchildId = None

        # 优化信息
        self.flag = None  # 这个是准备合并的时候用来标记的
        self.elements = []  # beam search得到的备份的元素[元素的ID，元素出现的次数]
        self.items = []  # beam search得到的备份的元素[元素ID，叶子节点的ID，在Knn中出现的次数]


class BinaryTree:
    """二叉树

    """

    def __init__(self):
        self.root = None
        self.nodes = []  # 树中的所有节点，保持空列表，使用时调用postOrder,用完清空
        self.leafNodes = []  # 树中所有的叶子节点
        self.recursion = 0  # 记录递归次数，同时给树的节点编号

    def TDcluster(self, k, node, pointsIndex, matrix, iteration=15, cluster_size=6):
        """

        :param k: k是最近邻的个数，树结构与k有关
        :param node: 树的节点，开始时为树的根节点
        :param pointsIndex: 需要分类的点的索引
        :param matrix: 分类点索引对应的具体坐标
        :param iteration: Kmeans的最大迭代次数，默认为 15
        :return:
        """

        matrix = np.asarray(matrix)

        if len(pointsIndex) < cluster_size * k:
            pointNum, cpoint = ut.CentralPointOfCluster(pointsIndex, matrix)
            leafNode = Node(cpoint, pointsIndex)
            leafNode.pointnum = pointNum
            self.leafNodes.append(leafNode)

            leafNode.Id = node.Id
            if node.father is None:
                leafNode.father = node.father
                leafNode.fatherId = node.Id
            else:
                leafNode.father = node.father
                leafNode.fatherId = node.father.Id

            return leafNode

        else:
            """
            这里找相对距离最远的中心点的方式为：找到未被划分的簇的中心点，找到距离中心点最远的数据点1，
            再找到距离数据点1最远的数据点2，该方法的时间复杂度为3n
            """
            # 随机给出两个两个簇的中心点
            cluster1 = matrix[0]
            cluster2 = matrix[0]

            # 找出中心点
            pointNum, cpoint = ut.CentralPointOfCluster(pointsIndex, matrix)

            # 找出相距中心点最远的点，即找到数据点1
            maxDistance = -float("inf")
            for ponint1Index in pointsIndex:
                v1 = matrix[ponint1Index]
                v2 = cpoint
                distance = ut.Compute_Euclidean(v1, v2)
                if maxDistance < distance:
                    maxDistance = distance
                    cluster1 = v1

            # 找出相距数据点1最远的数据点2，即找到相对最远的两个点
            maxDistance = -float("inf")
            for ponint1Index in pointsIndex:
                v2 = matrix[ponint1Index]
                v1 = cluster1
                distance = ut.Compute_Euclidean(v1, v2)
                if maxDistance < distance:
                    maxDistance = distance
                    cluster2 = v2

            # 给定聚类的中心点，即相对最远的两个数据点，有效降低迭代次数，优化聚类效果
            cpoints = []
            cpoints.append(cluster1)
            cpoints.append(cluster2)

            result = ut.Kmeans(cpoints, pointsIndex, matrix, iteration=iteration)

            self.recursion += 1
            node1 = Node(result[0][0])
            node1.pointnum = len(result[0][1])
            node1.Id = self.recursion
            node1.father = node
            node1.fatherId = node.Id

            self.recursion += 1
            node2 = Node(result[1][0])
            node2.pointnum = len(result[1][1])
            node2.Id = self.recursion
            node2.father = node
            node2.fatherId = node.Id

            node.lchild = self.TDcluster(k, node1, result[0][1], matrix)
            node.rchild = self.TDcluster(k, node2, result[1][1], matrix)

            node.lchildId = node1.Id
            node.rchildId = node2.Id

            return node

    def search_C(self, point, c=1):
        """ 在树中选择c枝查找结果

        :param point: 待查找的点
        :param c: 在树中查找的枝数
        :return: 候选点的集合, 叶子节点集合
        """

        # 防止叶子节点数小于遍历的枝数
        if len(self.leafNodes) < c:
            c = len(self.leafNodes)

        result = []
        queue = []
        queue.append(self.root)

        while True:

            # 踢掉队列中的元素，把它的孩子节点添加进来
            for snode in queue:
                if snode.lchild is not None:
                    queue.append(snode.lchild)
                    if snode in queue:
                        queue.remove(snode)

                if snode.rchild is not None:
                    queue.append(snode.rchild)
                    if snode in queue:
                        queue.remove(snode)

            # 存储queue中点到查询点的距离
            Distance = []
            if len(queue) < c:
                # 当查到的节点小于查询的枝数的时候，不做排序处理，继续添加节点，有效避免无效排序
                continue

            elif len(queue) >= c:
                v1 = np.asarray(point)

                for snode in queue:
                    v2 = snode.cpoint
                    distance = ut.Compute_Euclidean(v1, v2)
                    Distance.append(distance)

                DistanceOriginal = Distance.copy()
                # print(Distance)
                Distance.sort()
                QueueOriginal = queue.copy()
                for i in range(len(Distance)):
                    index = DistanceOriginal.index(Distance[i])
                    queue[i] = QueueOriginal[index]
                    DistanceOriginal[index] = -1

            # 保留距离查询点最近的c个节点（不一定全是叶子节点）
            queue = queue[:c]

            # 当queue中所有节点都为叶子节点时，查询结束
            sum_unleafs = 0
            for snode in queue:
                if snode.pointIndex is None:
                    sum_unleafs += 1
            if c - sum_unleafs == c:
                for lnode in queue:
                    result.extend(lnode.pointIndex)
                break

        result = list(set(result))

        return result, queue

    def postOrder(self, node):
        """ 后序遍历，不输出任何信息，而是将整个树的节点都存在nodes中

        :param node: 树的节点，开始时是树的根节点
        :return:
        """

        if node is None:
            return

        self.postOrder(node.lchild)
        self.postOrder(node.rchild)

        self.nodes.append(node)

    def SaveTree(self, filename):

        # 这里需要清空是害怕 postOrder 曾被多次调用过
        self.nodes = []
        self.postOrder(self.root)
        simple = Node([])
        nodeList = [simple for i in range(self.recursion + 1)]

        for node in self.nodes:
            if not nodeList[node.Id].cpoint:
                nodeList[node.Id] = node
            else:
                print("出现重复节点，代码有误")

        file = open(filename, "w")
        for node in nodeList:
            file.write("{}#".x(node.Id))  # 0
            file.write("{}#".format(node.fatherId))  # 1
            file.write("{}#".format(node.lchildId))  # 2
            file.write("{}#".format(node.rchildId))  # 3
            file.write("{}#".format(node.cpoint))  # 4
            file.write("{}#".format(node.pointIndex))  # 5
            file.write("{}#".format(node.elements))  # 6
            file.write("{}#".format(node.items))  # 7
            file.write("{}#".format(node.pointnum))  # 8
            file.write("{}\n".format(node.flag))  # 9

        file.close()
        self.nodes = []

    def LoadTree(self, fileName):

        nodeList = []
        with open(fileName, "r") as file:
            nodeMatrix = file.readlines()

        for nodestr in nodeMatrix:

            # 用 "#" 将节点的各个属性分离出来
            elementsStr = nodestr[:-1].split("#")

            nodeId = -1  # 点的 ID
            nodeFatherId = None  # 点的父节点的 ID
            nodeLchildId = None  # 点的左孩子节点的 ID
            nodeRchildId = None  # 点的右孩子节点的 ID
            nodeCpoint = []  # 中心点
            nodePointIndex = []  # 非叶子节点为 None，叶子节点存储的是类中点的索引
            nodeElements = []  # [[重复元素 Id， 出现次数], [], ...]
            nodeItems = []  # [[叶子节点 Id，叶子节点出现次数], ...]
            nodePointnum = None  # 用来标记当前节点的子树一共包含多少个点
            nodeFlag = None  # 这个是准备合并的时候用来标记的

            for i in range(len(elementsStr)):
                if i == 0:
                    # 点的ID
                    nodeId = int(elementsStr[i])

                elif i == 1:
                    # 点的父节点的ID
                    if elementsStr[i] == "None":
                        nodeFatherId = None
                    else:
                        nodeFatherId = int(elementsStr[i])

                elif i == 2:
                    # 点的左孩子节点的ID
                    if elementsStr[i] == "None":
                        nodeLchildId = None
                    else:
                        nodeLchildId = int(elementsStr[i])

                elif i == 3:
                    # 点的右孩子节点的ID
                    if elementsStr[i] == "None":
                        nodeRchildId = None
                    else:
                        nodeRchildId = int(elementsStr[i])

                elif i == 4:
                    # 中心点
                    strNumList = elementsStr[i][1:-1].split(",")
                    for num in strNumList:
                        num.strip()
                        nodeCpoint.append(float(num))

                elif i == 5:
                    # 非叶子节点为None，叶子节点存储的是类中点的索引
                    if elementsStr[i] == "None":
                        nodePointIndex = None
                    else:
                        strNumList = elementsStr[i][1:-1].split(",")
                        for num in strNumList:
                            num.strip()
                            if len(num) > 0:
                                nodePointIndex.append(int(num))

                elif i == 6:
                    # [[重复元素 Id， 出现次数], [], ...]
                    if elementsStr[i] == "[]":
                        nodeElements = []
                    else:
                        strTuples = elementsStr[i][1:-1].split("]")
                        for segment in strTuples:
                            element = []
                            if len(segment) >= 1:
                                if segment[0] == ",":
                                    segment = segment[3:]
                                elif segment[0] == "[":
                                    segment = segment[1:]
                            else:
                                # 这里说明一下，连续两个]]用“]”来分割的话，会出现一个空字符即""，长度为0
                                # print("出现不明物体'{}'".format(segment))
                                continue

                            for strNum in segment.split(","):
                                strNum = strNum.strip()
                                element.append(int(strNum))
                            nodeElements.append(element)

                elif i == 7:
                    # [[元素ID，在Knn中出现的次数，叶子节点的ID], ...]
                    if elementsStr[i] == "[]":
                        nodeItems = []
                    else:
                        strTuples = elementsStr[i][1:-1].split("]")
                        for segment in strTuples:
                            item = []
                            if len(segment) >= 1:
                                if segment[0] == ",":
                                    segment = segment[3:]
                                elif segment[0] == "[":
                                    segment = segment[1:]
                            else:
                                # print("出现不明物体'{}'".format(segment))
                                continue

                            for strNum in segment.split(","):
                                strNum = strNum.strip()
                                item.append(int(strNum))
                            nodeItems.append(item)

                elif i == 8:
                    # 用来标记当前节点的子树一共包含多少个点
                    nodePointnum = int(elementsStr[i])

                elif i == 9:
                    # 这个是准备合并的时候用来标记的
                    if elementsStr[i] == "None":
                        nodeFlag = None
                    else:
                        nodeFlag = int(elementsStr[i])

            node = Node(nodeCpoint)
            node.Id = nodeId
            node.fatherId = nodeFatherId
            node.lchildId = nodeLchildId
            node.rchildId = nodeRchildId
            node.pointIndex = nodePointIndex
            node.elements = nodeElements
            node.items = nodeItems
            node.pointnum = nodePointnum
            node.flag = nodeFlag

            nodeList.append(node)

        for node in nodeList:
            if node.Id == 0:
                if node.lchildId is None:
                    node.lchild = None
                else:
                    node.lchild = nodeList[node.lchildId]

                if node.rchildId is None:
                    node.rchild = None
                else:
                    node.rchild = nodeList[node.rchildId]

            else:
                if node.lchildId is None:
                    node.lchild = None
                else:
                    node.lchild = nodeList[node.lchildId]

                if node.rchildId is None:
                    node.rchild = None
                else:
                    node.rchild = nodeList[node.rchildId]

                node.father = nodeList[node.fatherId]

        self.root = nodeList[0]
        self.nodes = []
        self.postOrder(self.root)
        self.recursion = len(self.nodes) - 1
        for node in self.nodes:
            if node.pointIndex is not None:
                self.leafNodes.append(node)
        self.nodes = []

    def TreeInformation(self, node):
        # 这里是计算树的深度

        leftDeep = 0
        rightDeep = 0

        if node is None:
            return 0
        else:
            leftDeep = self.TreeInformation(node.lchild)
            rightDeep = self.TreeInformation(node.rchild)

            return max(leftDeep, rightDeep) + 1


class ElementsTree(BinaryTree):
    """增加重复元素的二叉树,即增加了数组
    """

    def __init__(self):
        super(ElementsTree, self).__init__()
        self.root = None
        self.nodes = []
        self.leafNodes = []  # 保存叶子节点
        self.recursion = 0  # 给节点编号

    def searchElement_C(self, point, c=1):
        """ 在树中选择c枝查找结果

        :param point: 待查找的点
        :param c: 在树中查找的枝数
        :return: 候选点的集合, 叶子节点集合

        """

        # 防止叶子节点数小于遍历的枝数
        if len(self.leafNodes) < c:
            c = len(self.leafNodes)

        result = []
        queue = [self.root]

        while True:

            for snode in queue:
                if snode.lchild is not None:
                    queue.append(snode.lchild)
                    if snode in queue:
                        queue.remove(snode)

                if snode.rchild is not None:
                    queue.append(snode.rchild)
                    if snode in queue:
                        queue.remove(snode)

            # 存储queue中点到查询点的距离
            Distance = []

            if len(queue) < c:
                continue
            elif len(queue) >= c:
                v1 = np.asarray(point)
                for snode in queue:
                    v2 = snode.cpoint
                    distance = ut.Compute_Euclidean(v1, v2)
                    Distance.append(distance)

                DistanceOriginal = Distance.copy()
                Distance.sort()
                QueueOriginal = queue.copy()
                for i in range(len(Distance)):
                    index = DistanceOriginal.index(Distance[i])
                    queue[i] = QueueOriginal[index]
                    DistanceOriginal[index] = -1

            # 保留距离查询点最近的c个节点（不一定全是叶子节点）
            queue = queue[:c]

            # 当queue中所有节点都为叶子节点时，查询结束
            sum_unleafs = 0
            for snode in queue:
                if snode.pointIndex is None:
                    sum_unleafs += 1
            if c - sum_unleafs == c:
                for lnode in queue:
                    result.extend(lnode.pointIndex)
                    # item 是 [元素Id，元素所在的叶子节点，出现的次数]，将数组中的元素也纳入考虑范围
                    for element in lnode.elements:
                        result.append(element[0])
                break

        result = list(set(result))

        return result, queue

    def ElementsOptimization(self, data_Matrix, optimizeMatrix, k=10, c=1, multiple=3, search=500, optimizeTruth=None):
        """

        :param data_Matrix: 数据点
        :param optimizeMatrix: 待优化的点
        :param optimizeTruth: 待优化的点的真实的Knn的集合
        :param k: 最近邻个数
        :param c: 搜索枝数
        :param multiple: 重复元素最大个数是最近邻个数的倍数
        :param search: 搜索枝数的倍数
        :return:
        """

        # 如果没有给出真实的Knn的结果，就用搜索search枝的结果来近似的代替真实结果
        if optimizeTruth is None:
            optimizeTruth = []
            for point in tqdm(optimizeMatrix):
                realCandid, nodesOfElement = self.search_C(point, search*c)
                real, RDistance = ut.Knn(k, point, realCandid, data_Matrix)
                optimizeTruth.append((real, RDistance))

        for i in tqdm(range(len(optimizeMatrix))):
            point = optimizeMatrix[i]

            # 预测的Knn的点
            predictCandid, leafNodes = self.search_C(point, c)
            predict, PDistance = ut.Knn(k, point, predictCandid, data_Matrix)

            real, RDistance = optimizeTruth[i]
            repeatElements = list(set(real) - set(predict))

            # 把所有的重复元素放入到涉及的叶子节点中
            for lNode in leafNodes:

                # 检查叶子的数组中是否已经包含该元素
                for repeatElement in repeatElements:
                    flag = True
                    minFrequency = float("inf")
                    for element in lNode.elements:

                        # 找到当前节点的最小频率
                        if minFrequency > element[1]:
                            minFrequency = element[1]

                        if repeatElement == element[0]:
                            element[1] += 1
                            flag = False
                            break

                    if flag:
                        if minFrequency == float("inf"):
                            minFrequency = 1
                        lNode.elements.append([repeatElement, minFrequency])

                elementsLen = multiple * k
                if len(lNode.elements) > elementsLen:

                    elementsOriginal = lNode.elements.copy()
                    frequencys = []
                    for element in lNode.elements:
                        frequencys.append(element[1])
                    frequencys.sort()

                    for j in range(len(elementsOriginal)):
                        index = frequencys.index(elementsOriginal[j][1])
                        lNode.elements[index] = elementsOriginal[j]
                        frequencys[index] = -1

                    lNode.elements = lNode.elements[-elementsLen:]


class ItemsTree(ElementsTree):
    """ 增加频繁项的二叉树

    """
    def __init__(self):
        super(ItemsTree, self).__init__()
        self.root = None
        self.nodes = []  # 树中的所有节点，保持空列表，使用时调用postOrder,用完清空
        self.leafNodes = []  # 树中所有的叶子节点
        self.recursion = 0  # 记录递归次数，同时给树的节点编号

    def ItemsOptimization(self, data_Matrix, optimizeMatrix, k=10, c=1, search=500, optimizeTruth=None):
        """
        :param data_Matrix: 训练集
        :param optimizeMatrix: 待优化的点
        :param optimizeTruth: 待优化的点的真实的Knn的集合
        :param k: 最近邻个数
        :param c: 搜索枝数
        :param multiple: 重复元素最大个数是最近邻个数的倍数
        :param search: 搜索枝数的倍数
        :return:
        """

        # 如果没有给出真实的Knn的结果，就用搜索search枝的结果来近似的代替真实结果
        if optimizeTruth is None:
            optimizeTruth = []
            for point in tqdm(optimizeMatrix):
                realCandid, nodesOfItem = self.search_C(point, search * c)
                real, RDistance = ut.Knn(k, point, realCandid, data_Matrix)
                optimizeTruth.append((real, RDistance, nodesOfItem))


        for i in range(len(optimizeMatrix)):
            point = optimizeMatrix[i]

            # 预测的Knn的点
            predictCandid, leafNodes = self.search_C(point, c)

            # optimizeTruth是真实的Knn或者是搜索枝数c为 500 时近似的Knn
            real, RDistance, nodesOfItem = optimizeTruth[i]

            # 将聚簇分类得到的Knn也加入优化的节点中
            nodesOfItem.extend(leafNodes)

            # 把所有参加 Knn 的元素放入到涉及的叶子的Item中
            for lNode in leafNodes:

                # 检查叶子的Item中是否已经包含该元素
                for pointIndex in real:
                    flag = True
                    for item in lNode.items:

                        if pointIndex == item[0]:
                            item[2] += 1
                            flag = False
                            break

                    # 如果叶子没有包含Item，则将其放入
                    if flag:
                        leafIdOfItem = -1
                        for node in nodesOfItem:
                            if pointIndex in node.pointIndex:
                                leafIdOfItem = node.Id
                                break

                        lNode.items.append([pointIndex, leafIdOfItem, 1])

    def RepartionOptimization(self, data_Matrix, file_name, k=10, cluster_size=6):

        # 这里需要清空是害怕 postOrder 曾被多次调用过
        self.nodes = []
        self.postOrder(self.root)
        simple = Node([])
        nodeList = [simple for i in range(self.recursion + 1)]

        for node in self.nodes:
            if not nodeList[node.Id].cpoint:
                nodeList[node.Id] = node
            else:
                print("出现重复节点，代码有误")

        # 构建相关性矩阵
        leafWithItems = []
        for leaf in self.leafNodes:
            if len(leaf.items) != 0:
                leafWithItems.append(leaf)

        # 取出叶子节点的Id
        leafId = []

        # Items不为空的叶子节点中包含的涉及 Knn 的点
        pointInKnn = []
        for leaf in leafWithItems:
            leafId.append(leaf.Id)

            for item in leaf.items:
                if item[0] in pointInKnn:
                    continue
                if item[0] in nodeList[item[1]].pointIndex:
                    index = nodeList[item[1]].pointIndex.index(item[0])
                    del nodeList[item[1]].pointIndex[index]
                pointInKnn.append(item[0])

        # Items不为空的叶子节点中包含的不涉及 Knn 的点
        pointNotInKnn = []
        for leaf in leafWithItems:
            for pointIndex in leaf.pointIndex:
                if pointIndex in pointInKnn:
                    continue
                else:
                    pointNotInKnn.append(pointIndex)

            # 清空元素节点的聚簇结果
            leaf.pointIndex = []

        # 构建二部图的矩阵
        matrix = []
        for i in range(len(leafWithItems)):
            weigt = [0 for _ in range(len(pointInKnn))]

            for item in leafWithItems[i].items:
                weigt[pointInKnn.index(item[0])] = item[2]

            for _ in range(cluster_size * k):
                matrix.append(weigt)

        # print(matrix)
        print(np.array(matrix).shape)

        # 计算最大匹配
        matrix = munkres.make_cost_matrix(
            matrix, lambda cost: sys.maxsize - cost if
            (cost != DISALLOWED) else DISALLOWED)
        indices = Munkres().compute(matrix)

        # 把最大匹配的结果写出来
        with open(file_name, "w") as f:
            for t in indices:
                f.write("{}\n".format(t))

        for leaf in leafWithItems:
            leaf.items = []

        # 重新分配涉及到 knn 的元素
        for repartitionKnn in indices:
            leafIndex, pointIndex = repartitionKnn
            leafIndex = int(leafIndex / (cluster_size * k))
            leafWithItems[leafIndex].pointIndex.append(pointInKnn[pointIndex])

        # 重新分配没涉及到 Knn 的节点
        for pointIndex in pointNotInKnn:
            distance = []
            flag = True
            v1 = data_Matrix[pointIndex]
            for leaf in leafWithItems:
                v2 = leaf.cpoint
                distance.append(ut.Compute_Euclidean(v1, v2))

            distanceCopy = distance.copy()
            distanceCopy.sort()
            for _ in distanceCopy:
                index = distance.index(_)
                if len(leafWithItems[index].pointIndex) < cluster_size * k:
                    leafWithItems[index].pointIndex.append(pointIndex)
                    flag = False
                    break

            if flag:
                index = distance.index(distanceCopy[0])
                leafWithItems[index].pointIndex.append(pointIndex)

    def IncrementalRepartion(self, data_Matrix, file_name, k=10, cluster_size=6, portion_size =4000):
        """

        :param data_Matrix: 数据的坐标点
        :param file_name:
        :param k: 最近邻个数
        :param cluster_size: 重复元素最大个数是最近邻个数的倍数
        :param portion_size:
        :return:
        """

        save_name = ""
        dir_list = file_name.split("/")
        for i in range(len(dir_list) - 1):
            save_name += dir_list[i] + "/"

        # 登录数据集
        dataMatrix, trainDataset, trainKnn = ut.LoadDataset(file_name)
        portion_indexes = random.sample(range(0, int(len(trainDataset) / portion_size)), int(0.4 * len(trainDataset) / portion_size))

        # 记录曾经出现过的Knn
        index_dictionary = {}

        for i in range(len(portion_indexes)):
            # 登录数据集
            dataMatrix, trainDataset, trainKnn = ut.LoadDataset(file_name)

            portion_index = portion_indexes[i]
            start_train = portion_index * portion_size
            if start_train + portion_size > len(trainDataset):
                end_train = len(trainDataset)
            else:
                end_train = start_train + portion_size

            trainDataset = trainDataset[start_train:end_train]

            print("第 {} 轮".format(i))

            # 激活elements优化条件，目前追求的是最高的准确率，所以表示为任意时候都激活
            thea = 0
            # for index in range(testMatrix.shape[0]):
            #     KnnIndex, KnnDistance = testTruth[index]
            #     thea += KnnDistance[thea_num]
            # thea = thea / testMatrix.shape[0]

            # 需要优化的查询点
            optimizeMatrix = []
            for index in range(trainDataset.shape[0]):

                point = trainDataset[index]
                # 为了平衡时间，所以搜索枝数为1
                candid, _ = self.search_C(point, 1)
                predict, predictDistance = ut.Knn(k, point, candid, dataMatrix)

                predictThea = 0
                for distance in predictDistance:
                    predictThea += distance

                if len(predictDistance) > 0:
                    predictThea = predictThea / len(predictDistance)

                if predictThea > thea or len(predictDistance) < k:
                    optimizeMatrix.append(point)

            self.ItemsOptimization(dataMatrix, optimizeMatrix)

            # 这里需要清空是害怕 postOrder 曾被多次调用过
            self.nodes = []
            self.postOrder(self.root)
            simple = Node([])
            nodeList = [simple for i in range(self.recursion + 1)]

            for node in self.nodes:
                if not nodeList[node.Id].cpoint:
                    nodeList[node.Id] = node
                else:
                    print("出现重复节点，代码有误")

            # 构建相关性矩阵
            leafWithItems = []
            for leaf in self.leafNodes:
                if len(leaf.items) != 0:
                    leafWithItems.append(leaf)

            # 取出叶子节点的Id
            leafId = []

            # Items不为空的叶子节点中包含的涉及 Knn 的点
            pointInKnn = []
            for leaf in leafWithItems:
                leafId.append(leaf.Id)

                for item in leaf.items:
                    if str(item[0]) in index_dictionary.keys():
                        index_dictionary[str(item[0])] += 1
                    else:
                        index_dictionary[str(item[0])] = item[2]

                    if item[0] in pointInKnn:
                        continue
                    if item[0] in nodeList[item[1]].pointIndex:
                        index = nodeList[item[1]].pointIndex.index(item[0])
                        del nodeList[item[1]].pointIndex[index]

                    pointInKnn.append(item[0])

            # Items不为空的叶子节点中包含的不涉及 Knn 的点
            pointNotInKnn = []
            for leaf in leafWithItems:
                for pointIndex in leaf.pointIndex:
                    if pointIndex in index_dictionary.keys():
                        continue
                    else:
                        pointNotInKnn.append(pointIndex)

                        # 清除掉不涉及到knn的结果
                        index = leaf.pointIndex.index(pointIndex)
                        del leaf.pointIndex[index]

                # 清空元素节点的聚簇结果
                # leaf.pointIndex = []

            # 构建二部图的矩阵
            matrix = []
            for i in range(len(leafWithItems)):
                weigt = [0 for _ in range(len(pointInKnn))]

                for item in leafWithItems[i].items:

                    weigt[pointInKnn.index(item[0])] = index_dictionary[str(item[0])]

                for _ in range(cluster_size * k):
                    matrix.append(weigt)

            # print(matrix)
            print(np.array(matrix).shape)

            # 计算最大匹配
            matrix = munkres.make_cost_matrix(
                matrix, lambda cost: sys.maxsize - cost if
                (cost != DISALLOWED) else DISALLOWED)
            indices = Munkres().compute(matrix)

            # 把最大匹配的结果写出来
            with open(save_name + "indices.txt", "w") as f:
                for t in indices:
                    f.write("{}\n".format(t))

            # 清空叶子节点中的items
            for leaf in leafWithItems:
                leaf.items = []


            # 重新分配涉及到 knn 的元素
            for repartitionKnn in indices:
                leafIndex, pointIndex = repartitionKnn
                leafIndex = int(leafIndex / (cluster_size * k))

                if pointInKnn[pointIndex] in leafWithItems[leafIndex].pointIndex:
                    continue

                leafWithItems[leafIndex].pointIndex.append(pointInKnn[pointIndex])

            # 重新分配没涉及到 Knn 的节点
            for pointIndex in pointNotInKnn:
                distance = []
                flag = True
                v1 = data_Matrix[pointIndex]
                for leaf in leafWithItems:
                    v2 = leaf.cpoint
                    distance.append(ut.Compute_Euclidean(v1, v2))

                distanceCopy = distance.copy()
                distanceCopy.sort()
                for _ in distanceCopy:
                    index = distance.index(_)
                    if len(leafWithItems[index].pointIndex) < cluster_size * k:
                        leafWithItems[index].pointIndex.append(pointIndex)
                        flag = False
                        break

                if flag:
                    index = distance.index(distanceCopy[0])
                    leafWithItems[index].pointIndex.append(pointIndex)

