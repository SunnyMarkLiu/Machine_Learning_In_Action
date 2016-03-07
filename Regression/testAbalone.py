#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
利用标准线性规划和局部加权线性规划预测鲍鱼年龄
@Author: MarkLiu
"""

from regression import *
import numpy as np
import matplotlib.pyplot as plt


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
    datasArr, valuesArr = loadDataSet('datasets/abalone.txt')
    ws = standardRegression(datasArr, valuesArr)
    testDatas, testValues = loadDataSet('datasets/testAbalone.txt')

    testMat = np.matrix(testDatas)
    m = np.shape(testDatas)[0]
    predictValues = []
    for i in range(0, m):
        predictValue = float(testMat[i] * ws)
        predictValues.append(predictValue)
        print '预测值：', float(predictValue), "实际值：", testValues[i]
    error = calError(predictValues, testValues)
    print "均方误差为：", error


def testLocallyWeightedRegression():
    datasArr, valuesArr = loadDataSet('datasets/abalone.txt')
    testDatas, testValues = loadDataSet('datasets/testAbalone.txt')

    m = np.shape(testDatas)[0]
    predictValues = []
    for i in range(0, m):
        predictValue = locallyWeightedRegression(testDatas[i], datasArr, valuesArr, 0.1)
        predictValues.append(float(predictValue))
        print '预测值：', float(predictValue), "实际值：", testValues[i]
    error = calError(predictValues, testValues)
    print "均方误差为：", error


def testRidgeRegression():
    """
    测试岭回归
    :return:
    """
    datasArr, valuesArr = loadDataSet('datasets/abalone.txt')
    datasMat = np.matrix(datasArr)
    valuesMat = np.matrix(valuesArr).T

    # rumor输得样本数比数据的特征少，需要将数据标准化，压缩系数理解数据
    valuesMean = np.mean(valuesMat, 0)   # 计算矩阵指定轴的平均值
    datasMean = np.mean(datasMat, 0)
    datasVar = np.var(datasMat, 0)

    datasMat = (datasMat - datasMean) / datasVar
    valuesMat = valuesMat - valuesMean

    ws = ridgeRegression(datasMat, valuesMat, lamb=0.2)
    print ws
    testDatas, testValues = loadDataSet('datasets/testAbalone.txt')

    testMat = np.matrix(testDatas)
    m = np.shape(testDatas)[0]
    predictValues = []
    for i in range(0, m):
        predictValue = float(testMat[i] * ws)
        predictValues.append(predictValue)
        print '预测值：', float(predictValue), "实际值：", testValues[i]
    error = calError(predictValues, testValues)
    print "均方误差为：", error


def testRidgeRegressionLambda():
    """
    测试岭回归
    :return:
    """
    datasArr, valuesArr = loadDataSet('datasets/abalone.txt')
    datasMat = np.matrix(datasArr)
    valuesMat = np.matrix(valuesArr).T

    # rumor输得样本数比数据的特征少，需要将数据标准化，压缩系数理解数据
    valuesMean = np.mean(valuesMat, 0)   # 计算矩阵指定轴的平均值
    datasMean = np.mean(datasMat, 0)
    datasVar = np.var(datasMat, 0)

    datasMat = (datasMat - datasMean) / datasVar
    valuesMat = valuesMat - valuesMean

    numTest = 30
    wsMat = np.zeros((numTest, np.shape(datasMat)[1]))
    for i in range(0, numTest):
        ws = ridgeRegression(datasMat, valuesMat, np.exp(i-10))
        wsMat[i, :] = ws.T
    print wsMat
    # 绘制Lambda系数对ws的变化情况
    plt.figure(figsize=(10, 10), facecolor="white")
    plt.subplot(111)
    print "wsMat", np.shape(wsMat)
    plt.plot(wsMat)
    plt.show()

if __name__ == '__main__':
    # testStandardRegression()
    # testLocallyWeightedRegression()
    # testRidgeRegression()
    testRidgeRegressionLambda()
