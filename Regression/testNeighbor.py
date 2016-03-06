#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""

from regression import *


def calError(predictValuesArr, testValues):
    """
    计算均方误差
    :param predictValuesArr:
    :param testValues:
    :return:
    """
    predictValuesMat = np.matrix(predictValuesArr)
    testValues = np.matrix(testValues)
    gaps = predictValuesMat - testValues
    return np.sqrt((gaps * gaps.T).sum())


dataSet = loadAllDataSet('datasets/abalone.txt')

testDatas, valuessArr = loadDataSet('datasets/testAbalone.txt')
diem = chooseBestFeatureAxisToSort(dataSet)
sortedFeatures, sortedResult = sortDataSet(diem, dataSet)

predictValues = []
for i in range(0, len(testDatas)):
    predictValue = locallyWeightedRegressionNeighbor(testDatas[i], diem, sortedFeatures, sortedResult, 0.3)
    predictValues.append(float(predictValue))
    print "实际结果：", valuessArr[i], ",预测结果：", float(predictValue)

error = calError(predictValues, valuessArr)
print error
