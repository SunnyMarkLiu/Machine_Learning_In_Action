#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""

from regression import *
import numpy as np
import matplotlib.pyplot as plt


def testStandardRegression():
    datasArr, valuessArr = loadDataSet('datasets/ex0.txt')
    # 利用标准回归计算系数W
    ws = standardRegression(datasArr, valuessArr)

    # 绘制原始数据
    xMat = np.matrix(datasArr)
    valueMat = np.matrix(valuessArr)
    plt.figure(figsize=(10, 10), facecolor="white")
    plt.subplot(111)
    plt.scatter(xMat[:, 1].flatten().A[0], valueMat.T.flatten().A[0])

    # 绘制回归的曲线
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    # 绘制预测的曲线
    plt.plot(xCopy[:, 1], yHat)
    plt.show()

    # 计算预测值和实际值的相关性
    yHat = xMat * ws
    print np.shape(yHat), ":", np.shape(valueMat)
    # 相关性系数
    correlationCoefficients = np.corrcoef(yHat.T, valueMat)
    print "相关系数为", correlationCoefficients


def testLocallyWeightedRegression():
    datasArr, valuessArr = loadDataSet('datasets/ex0.txt')
    m = np.shape(datasArr)[0]
    predictValues = np.zeros(m)
    for i in range(0, m):
        predictValues[i] = \
            locallyWeightedRegression(datasArr[i], datasArr, valuessArr, 0.01)

    # 绘制原始数据
    xMat = np.matrix(datasArr)
    valueMat = np.matrix(valuessArr)
    plt.figure(figsize=(10, 10), facecolor="white")
    plt.subplot(111)
    plt.scatter(xMat[:, 1].flatten().A[0], valueMat.T.flatten().A[0])
    # 绘制回归的曲线
    # 先对测试数据进行排序
    sortedIndexs = xMat[:, 1].argsort(0)
    print "sortedIndexs:"
    print sortedIndexs
    sortedMat = xMat[sortedIndexs.flatten().A[0]]
    plt.plot(sortedMat[:, 1], predictValues[sortedIndexs], c='red', linewidth=2)
    plt.show()
    # 计算预测值和实际值的相关性
    correlationCoefficients = np.corrcoef(predictValues, valueMat)
    print "相关系数为", correlationCoefficients


def testLocallyWeightedRegressionNeighbor():
    dataSet = loadAllDataSet('datasets/ex0.txt')
    testDatas, valuessArr = loadDataSet('datasets/ex0.txt')
    diem = chooseBestFeatureAxisToSort(dataSet)
    sortedFeatures, sortedResult = sortDataSet(diem, dataSet)
    m = np.shape(testDatas)[0]
    predictValues = np.zeros(m)
    for i in range(0, m):
        predictValue = locallyWeightedRegressionNeighbor(testDatas[i], diem, sortedFeatures, sortedResult, 0.01)
        predictValues[i] = float(predictValue)

    # 绘制原始数据
    xMat = np.matrix(testDatas)
    valueMat = np.matrix(valuessArr)
    plt.figure(figsize=(10, 10), facecolor="white")
    plt.subplot(111)
    plt.scatter(xMat[:, 1].flatten().A[0], valueMat.T.flatten().A[0])
    # 绘制回归的曲线
    # 先对测试数据进行排序
    sortedIndexs = xMat[:, 1].argsort(0)
    print "sortedIndexs:"
    print sortedIndexs
    sortedMat = xMat[sortedIndexs.flatten().A[0]]
    plt.plot(sortedMat[:, 1], predictValues[sortedIndexs], c='red', linewidth=2)
    plt.show()
    # 计算预测值和实际值的相关性
    correlationCoefficients = np.corrcoef(predictValues, valueMat)
    print "相关系数为", correlationCoefficients

if __name__ == '__main__':
    # testStandardRegression()
    # testLocallyWeightedRegression()
    testLocallyWeightedRegressionNeighbor()
