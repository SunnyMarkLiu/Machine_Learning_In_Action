#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""
import numpy as np


def loadDataSet(fileName):
    """
    加载数据
    :param fileName:
    :return:
    """
    dataArr = []
    fr = open(fileName)
    for line in fr.readlines():
        datas = line.strip().split('\t')
        # 批量转换sequence数据
        floatDatas = map(float, datas)
        dataArr.append(floatDatas)
    return dataArr


def calcEuclideanDistance(vectorA, vectorB):
    """
    计算欧式距离
    :param vectorA:
    :param vectorB:
    :return:
    """
    return np.sqrt(np.sum(np.power(vectorA - vectorB, 2)))


def initialCentralPoint(dataSet, k):
    """
    初始化随机产生K个簇的质心
    :param dataSet:
    :param k:
    :return:
    """
    dataMat = np.matrix(dataSet)
    n = np.shape(dataSet)[1]
    clusterPoints = np.matrix(np.zeros((k, n)))
    for j in range(n):
        minFeatureJ = min(dataMat[:, j])  # 获取第j列特征的最小值
        rangeFeatureJ = float(max(dataMat[:, j]) - minFeatureJ)
        # 计算所有簇的第J个特征的值
        clusterPoints[:, j] = minFeatureJ + rangeFeatureJ * np.random.rand(k, 1)
    return clusterPoints


def kMeans(dataSet, k, distMeasure=calcEuclideanDistance,
           initCentral=initialCentralPoint):
    """
    K-均值算法
    :param dataSet:
    :param k:
    :param distMeasure:
    :param initCentral:
    :return:
    """
    dataSet = np.matrix(dataSet)
    m = np.shape(dataSet)[0]
    # 保存所有数据点所在簇的下标及对应的距离
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 初始化簇的质心
    centroids = initCentral(dataSet, k)
    # centroids = np.matrix([[-4.34827349, 1.37075232],
    #                        [3.21097122, -2.64897986],
    #                        [0.59106467, 4.6343214],
    #                        [3.73812572, 0.61857479]])
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 遍历所有的数据
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                # 计算质心和数据点的距离
                distJI = distMeasure(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            # 保存距离最小的簇下标和对应的距离
            clusterAssment[i, :] = minIndex, minDist ** 2
        print centroids
        print "---"
        for cent in range(k):  # 更新簇的质心
            # clusterAssment的第一列保存对应数据所属的簇
            currentClusters = clusterAssment[:, 0].A
            # 返回非零元素的索引
            nonZeroIndex = np.nonzero(currentClusters == cent)[0]
            ptsInClust = dataSet[nonZeroIndex]  # 获取所有在同一cluster的数据
            centroids[cent, :] = np.mean(ptsInClust, axis=0)  # 在列轴计算平均值
    return centroids, clusterAssment
