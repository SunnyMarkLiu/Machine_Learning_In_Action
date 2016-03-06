#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""
import numpy as np
import random


def loadDataSet(fileName):
    """
    可根据数据集的特征的个数自适应加载数据
    :param fileName:
    :return:
    """
    frTrain = open(fileName)
    datasArr = []
    valuessArr = []
    for line in frTrain.readlines():
        linedatas = line.strip().split('\t')
        floatdatas = []
        for i in range(len(linedatas)):
            floatdatas.append(float(linedatas[i]))
        datasArr.append(floatdatas[:-1])
        valuessArr.append(float(floatdatas[-1]))

    frTrain.close()
    return datasArr, valuessArr


def loadAllDataSet(fileName):
    """
    可根据数据集的特征的个数自适应加载数据
    :param fileName:
    :return:
    """
    frTrain = open(fileName)
    dataSet = []
    for line in frTrain.readlines():
        linedatas = line.strip().split('\t')
        datas = []
        for i in range(0, len(linedatas)):
            datas.append(float(linedatas[i]))
        dataSet.append(datas)

    frTrain.close()
    return dataSet


def standardRegression(datasArr, valuessArr):
    """
    最基本的线性规划，求系数矩阵W
    :param datasArr:
    :param valuessArr:
    :return:
    """
    # 保证数据为numpy矩阵
    xMat = np.matrix(datasArr)
    valueMat = np.matrix(valuessArr).T
    # W = (xTx)^(-1)*xTy
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:  # 如果xTx矩阵的行列式为0,则不存在逆矩阵
        print "xTx矩阵的行列式为0,则不存在逆矩阵!"

    ws = xTx.I * (xMat.T * valueMat)
    return ws


def locallyWeightedRegression(testPoint, datasArr, valuessArr, k=1):
    """
    局部加权线性回归
    :param testPoint: 测试的数据点
    :param datasArr: 样本数据
    :param valuessArr: 样本数据的结果
    :param k: 控制权重下降的速度，k越小下降速度越快
    :return:
    """
    # 保证数据为numpy矩阵
    xMat = np.matrix(datasArr)
    valueMat = np.matrix(valuessArr).T

    # 为每个样本舒适化一个权重
    m = np.shape(xMat)[0]
    # 创建一个对角矩阵，对角线上为1
    weights = np.matrix(np.eye(m))
    """
    此处可以看出局部加权线性回归存在一个问题，它对每个点做预测时都计算整个数据集
    和测试数据的差值以确定权重，使计算量大大增加；而且距离测试数据越远的样本数据
    ，其权重越小，也会影响精度。可以设定一邻域，只计算邻域内样本数据的权重。减小
    计算量，同时提高精度。
    """
    for i in range(0, m):
        featureDiffMat = testPoint - xMat[i]  # 将测试数据特征值和样本的差别，结果为矩阵
        # 计算差别的平方根距离
        distance = featureDiffMat * featureDiffMat.T
        weights[i, i] = np.exp(distance / (-2 * k ** 2))
    xTx = xMat.T * weights * xMat
    if np.linalg.det(xTx) == 0.0:  # 如果xTx矩阵的行列式为0,则不存在逆矩阵
        print "xTx矩阵的行列式为0,则不存在逆矩阵!"

    ws = xTx.I * (xMat.T * weights * valueMat)
    predictValue = testPoint * ws
    return predictValue


def chooseBestFeatureAxisToSort(dataSet):
    """
    获取最佳排序的参考feature
    :param dataSet: 所有的样本数据集，包括结果值
    :return:    返回最佳分类标准的坐标diem
    """
    m, n = np.shape(dataSet)
    # 分别获取特征数据的列表和对应的结果
    featureSet = []
    resultsSet = []
    # for i in range(0, m):
    #     featureSet.append(dataSet[i][:-1])
    #     resultsSet.append(dataSet[i][-1])

    for i in range(0, n-1):
        randomIndex = int(random.uniform(0, m-1))
        featureSet.append(dataSet[randomIndex][:-1])
        resultsSet.append(dataSet[randomIndex][-1])

    featureMat = np.multiply(np.matrix(featureSet), 10)
    resultsMat = np.matrix(resultsSet).T
    print featureMat
    if np.linalg.det(featureMat) == 0.0:  # 如果xTx矩阵的行列式为0,则不存在逆矩阵
        print "featureMat矩阵的行列式为0,则不存在逆矩阵!"
    Wmn = featureMat.I * resultsMat
    diem = np.abs(Wmn).argmax()     # 应该将Wmn取绝对值，再取最大值
    return diem


def sortDataSet(diem, dataSet):
    """
    按照diem维特征排序数据集(待修改！！)
    :param diem:
    :param dataSet:
    :return:
    """
    datasMat = np.matrix(dataSet)
    sortedIndexs = datasMat[:, diem].argsort(0)
    sortedDataMat = datasMat[sortedIndexs.flatten().A[0]]
    # print datasMat
    m, n = np.shape(sortedDataMat)

    featureArr = []
    valuessArr = []
    for i in range(0, m):
        linedatas = sortedDataMat[i].tolist()[0]
        floatdatas = []
        for j in range(len(linedatas)):
            floatdatas.append(float(linedatas[j]))
        featureArr.append(floatdatas[:-1])
        valuessArr.append(float(floatdatas[-1]))

    return featureArr, valuessArr


def locallyWeightedRegressionNeighbor(testPoint, diem, sortedFeatures, sortedResult, neighborPercentage):
    """
    局部加权线性回归
    :param testPoint: 测试的数据点
    :param diem: 分类的特征下标
    :param sortedFeatures:
    :param sortedResult:
    :param neighborPercentage: 设置计算权重时，参考邻域内样本的数目占样本的百分比
    :return:
    """
    # 保证数据为numpy矩阵
    sortedFeaturesMat = np.matrix(sortedFeatures)

    # 为每个样本舒适化一个权重
    m = np.shape(sortedFeaturesMat)[0]
    # 确定testPoint的位置
    testIndex = 0
    sortedFeat = sortedFeaturesMat[:, diem].flatten().A[0]
    for i in range(0, m):
        if float(testPoint[diem]) > float(sortedFeat[i]):
            testIndex = i

    left = max(int(testIndex - neighborPercentage * m), 0)
    right = min(int(testIndex + neighborPercentage * m), m)

    calcFeature = []
    calcResults = []
    # 选择邻域数据
    for i in range(left, right):
        calcFeature.append(sortedFeatures[i])
        calcResults.append(sortedResult[i])

    calcFeatureMat = np.matrix(calcFeature)
    calcResultsMat = np.matrix(calcResults).T
    xTx = calcFeatureMat.T * calcFeatureMat
    if np.linalg.det(xTx) == 0.0:  # 如果xTx矩阵的行列式为0,则不存在逆矩阵
        print "xTx矩阵的行列式为0,则不存在逆矩阵!"

    ws = xTx.I * (calcFeatureMat.T * calcResultsMat)
    # print "权重系数："
    # print ws
    predictValue = testPoint * ws
    return predictValue
