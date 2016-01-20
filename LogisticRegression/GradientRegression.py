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
    featureDatasList = []
    classTypes = []
    datafile = open('dataset/testSet.data')
    allLines = datafile.readlines()
    for line in allLines:
        line = line.strip().split('\t')
        featureDatasList.append([1.0, float(line[0]), float(line[1])])
        classTypes.append(int(line[2]))

    return featureDatasList, classTypes


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
    :type featureDatas: list
    :type classTypes: list
    :param featureDatas: m x n的list，m个样本，n个特征
    :param classTypes: m个样本的类别
    :return:
    """
    print type(featureDatas), type(classTypes)
    print np.shape(featureDatas), np.shape(classTypes)
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
    :type featureDatas: list
    :type classTypes: list
    :param featureDatas: m x n的list，m个样本，n个特征
    :param classTypes: m个样本的类别
    :param weights:
    :return:
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc')

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
    class0 = plotaxes.scatter(class0X, class0Y, marker='o', s=40, c='red')
    class1 = plotaxes.scatter(class1X, class1Y, marker='s', s=20, c='green')
    # 绘制决策边界
    # z = W0X0 + W1X1 + W2X2 当Z=0时，sigmod行数的值为0.5,即判断类别的边界
    # 所以求得X2 = （-W0 -W1X1）/ W2
    X1 = np.linspace(-4, 4, 50, endpoint=True)
    print 'weights:', weights
    X2 = (-1 * weights[0] - weights[1] * X1) / weights[2]
    plotaxes.plot(X1, X2)
    plt.title(u'梯度上升算法/随机梯度上升算法绘制的最佳拟合曲线', fontproperties=zhfont, size=15)
    plt.xlabel(u'特征X1', fontproperties=zhfont, size=18)
    plt.ylabel(u'特征X2', fontproperties=zhfont, size=18)
    plotaxes.legend((class0, class1), ('class0', 'class1'))
    plt.show()


def getBestWeightsByRandomGradientAscent(featureDatasList, classTypes, maxCycles=1):
    """
    改进后的随机梯度上升算法获取最佳回归系数
    :type featureDatasList: list
    :type classTypes: list
    :param featureDatasList:
    :param classTypes:
    :param maxCycles: 设置最大迭代次数，默认为1
    :return:
    """
    import random

    featureDatas = np.array(featureDatasList)
    m, n = np.shape(featureDatas)
    # 梯度上升的步长
    weights = np.ones(n)
    # 保存每次循环计算的weights的结果
    weightsMatrix = []
    for j in range(maxCycles):
        featureIndexs = range(m)
        for i in range(m):  # 对每个样本数据进行遍历，计算更新回归系数
            delta = 1 / (1.0 + i + j) + 0.001
            randIndex = int(random.uniform(0, len(featureIndexs)))
            # 计算预测函数sigmod的输入：Z = W0X0 + W1X1 + ...
            sigmodInput = sum(featureDatas[randIndex] * weights)
            estimateClass = calculateSigmodEstimateClassType(sigmodInput)
            error = classTypes[randIndex] - estimateClass
            # 更新回归系数
            weights += (error * delta) * featureDatas[randIndex]
            del(featureIndexs[randIndex])
        weightsMatrix.append(weights.tolist())

    return weightsMatrix, weights


def plotWeightsAstringency(featureDatas, classTypes):
    """
    绘制采用随机梯度上升算法获取的回归参数随着迭代次数增加的收敛性
    :type featureDatas: list
    :type classTypes: list
    :param featureDatas: m x n的list，m个样本，n个特征
    :param classTypes: m个样本的类别
    :return:
    """
    import matplotlib.pyplot as plt
    # import matplotlib as mpl
    #
    # zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc')

    maxIteratCounts = 600
    num = range(maxIteratCounts)
    weightsMatrix, weights = getBestWeightsByRandomGradientAscent(featureDatas, classTypes, maxIteratCounts)

    figure = plt.figure(facecolor='white')
    # 将array转换为ndarray
    weightsMatrix = np.array(weightsMatrix)
    X0 = weightsMatrix[:, 0]
    X1 = weightsMatrix[:, 1]
    X2 = weightsMatrix[:, 2]
    print X1
    pltaxes0 = figure.add_subplot(311)
    pltaxes0.plot(num, X0, color="blue", linewidth=2.0, linestyle="-")
    pltaxes1 = figure.add_subplot(312)
    pltaxes1.plot(num, X1, color="blue", linewidth=2.0, linestyle="-")
    pltaxes2 = figure.add_subplot(313)
    pltaxes2.plot(num, X2, color="blue", linewidth=2.0, linestyle="-")
    # plt.title(u'随机梯度上升算法回归参数随迭代次数的收敛性', fontproperties=zhfont, size=15)
    # plt.xlabel(u'迭代次数', fontproperties=zhfont, size=18)
    # plt.ylabel(u'特征X1', fontproperties=zhfont, size=18)
    plt.show()


def logisticClassify(inputFeatures, trainWeights):
    """
    基于随机梯度上升的logistic回归分类器
    :param inputFeatures:
    :param trainWeights:
    :return:
    """
    z = sum(inputFeatures * trainWeights)
    prob = calculateSigmodEstimateClassType(z)
    if prob > 0.5:
        return 1.0
    else:
        return 0.0
