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
    print "xMat:", np.shape(xMat)
    valueMat = np.matrix(valuessArr).T
    print "valueMat:", np.shape(valueMat)
    # W = (xTx)^(-1)*xTy
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:   # 如果xTx矩阵的行列式为0,则不存在逆矩阵
        print "xTx矩阵的行列式为0,则不存在逆矩阵!"

    ws = xTx.I * (xMat.T * valueMat)
    return ws
