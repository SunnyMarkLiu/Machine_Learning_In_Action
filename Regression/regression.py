#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""
import numpy as np


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
