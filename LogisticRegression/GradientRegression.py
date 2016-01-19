#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""
import numpy as np


def loadDataSet():
    """
    加载数据
    :return:
    """
    featureDatas = []
    classTypes = []
    datafile = open('dataset/testSet.data')
    allLines = datafile.readlines()
    for line in allLines:
        line = line.strip().split('\t')
        featureDatas.append([1.0, float(line[0]), float(line[1])])
        classTypes.append(int(line[2]))

    return featureDatas, classTypes


def calculateSigmodEstimateClassType(x):
    """
    当输入为x时计算Sigmod函数的值，如果输入的是矩阵，则返回的也是矩阵
    :param x:
    :return:
    """
    # 此处由于classTypes数据的值为0,1 所以这个函数的系数为1,
    # 如果classTypes数据类型超过1，则对应的乘以相应的系数
    return 1.0 / (1 + np.exp(-1 * x))


def getBestRegressionWeightsByGradientAscent(featureDatas, classTypes):
    """
    梯度上升算法获取最佳回归系数
    :param featureDatas:
    :param classTypes:
    :return:
    """
    # 将python的list转换为NumPy的矩阵
    featureDatasMat = np.mat(featureDatas)
    classTypesMat = np.mat(classTypes).transpose()
    m, n = np.shape(featureDatasMat)
    # 梯度上升的步长
    delta = 0.01
    maxCycles = 500  # 最大循环次数
    # 定义回归系数，n表示和特征数据的列数相等
    weights = np.ones((n, 1))
    for i in range(maxCycles):
        # 按照找此系数乘以featureDatas，得到的估计classType为：
        estimateClasses = calculateSigmodEstimateClassType(featureDatasMat * weights)
        # 得到预测的误差，error为nx1的列向量
        error = classTypesMat - estimateClasses
        # 梯度上升调整系数，将每一组训练数据乘以对应的误差和步长，再加上原系数
        # 这也就说明了featureDatasMat需要转置
        weights += delta * featureDatasMat.transpose() * error

    return weights


def plotBestRegressionLine(featureDatas, classTypes, weights):
    """
    根据weights绘制最佳拟合曲线进行分类
    :param classTypes:
    :param featureDatas:
    :param weights:
    :return:
    """
    import matplotlib.pyplot as plt

    class0X = []
    class0Y = []
    class1X = []
    class1Y = []

    m = np.shape(featureDatas)[0]
    for i in range(m):
        if classTypes[i] == 0:  # 如果是类别0
            class0X.append(featureDatas[i][1])
            class0Y.append(featureDatas[i][2])
        else:  # 如果是类别1
            class1X.append(featureDatas[i][1])
            class1Y.append(featureDatas[i][2])

    figure = plt.figure(facecolor='white')
    plotaxes = figure.add_subplot(111)
    # 绘制两种不同类型的数据
    plotaxes.scatter(class0X, class0Y, marker='o', s=40, c='red')
    plotaxes.scatter(class1X, class1Y, marker='s', s=20, c='green')
    # 绘制决策边界
    # z = W0X0 + W1X1 + W2X2 当Z=0时，sigmod行数的值为0.5,即判断类别的边界
    # 所以求得X2 = （-W0 -W1X1）/ W2
    X1 = np.linspace(-4, 4, 50, endpoint=True)
    X2 = (-weights[0] - weights[1] * X1) / weights[2]
    plotaxes.plot(X1, X2)
    plt.show()
