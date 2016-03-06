#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
利用标准线性规划和局部加权线性规划预测鲍鱼年龄
@Author: MarkLiu
"""

from regression import *
import regression
import numpy as np


def calError(predictValues, testValues):
    """
    计算均方误差
    :param predictValues:
    :param testValues:
    :return:
    """
    predictValues = np.matrix(predictValues)
    testValues = np.matrix(testValues)
    gaps = predictValues - testValues
    print type(gaps), np.shape(gaps)
    print gaps
    return np.sqrt((gaps * gaps.T).sum())


def testStandardRegression():
    datasArr, valuessArr = loadDataSet('datasets/abalone.txt')
    ws = regression.standardRegression(datasArr, valuessArr)
    testDatas, testValues = loadDataSet('datasets/testAbalone.txt')

    testMat = np.matrix(testDatas)
    m = np.shape(testDatas)[0]
    predictValues = []
    for i in range(0, m):
        predictValue = float(testMat[i] * ws)
        predictValues.append(predictValue)
        print '预测值：', float(predictValue), "实际值：", testValues[i]
    error = calError(predictValues, testValues)
    print error


def testLocallyWeightedRegression():
    datasArr, valuessArr = loadDataSet('datasets/abalone.txt')
    testDatas, testValues = loadDataSet('datasets/testAbalone.txt')

    m = np.shape(testDatas)[0]
    predictValues = []
    for i in range(0, m):
        predictValue = locallyWeightedRegression(testDatas[i], datasArr, valuessArr, 0.1)
        predictValues.append(float(predictValue))
        print '预测值：', float(predictValue), "实际值：", testValues[i]
    error = calError(predictValues, testValues)
    print error

if __name__ == '__main__':
    # testStandardRegression()
    testLocallyWeightedRegression()
