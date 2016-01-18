#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
"""
import operator
from math import log
import pickle


# ----------- 分类数据集 -----------


def calculateShannonEntropy(dataset):
    """
    计算数据集的香农信息熵
    :param dataset:
    :return:
    """
    labelCountDitc = {}  # 记录label和出现的次数的dict
    for rowdata in dataset:  # 遍历每一行数据
        label = rowdata[-1]  # 当前行数据的label
        if label not in labelCountDitc.keys():
            labelCountDitc[label] = 0
        labelCountDitc[label] += 1  # 记录每个label出现的次数

    datasetNum = len(dataset)  # 待计算数据的总数目
    shannonEntropy = 0.0
    for label in labelCountDitc.keys():
        p = float(labelCountDitc[label]) / datasetNum  # 计算当前label出现的概率
        shannonEntropy -= p * log(p, 2)  # 利用公式计算香农信息熵

    return shannonEntropy


def chooseBestFeatureAxisToSplit(dataSet):
    """
    分析所有可能的分类效果，计算分类数据集的信息熵，选择熵最小的分类方法。
    上越校说明，分类的数据越'纯'，分类效果越好
    :param dataSet:
    :return:
    """
    dataSetLength = len(dataSet)
    splitedCount = len(dataSet[0]) - 1  # 可分类的特征数目
    # 未分类的原始数据的信息熵
    baseEntropy = calculateShannonEntropy(dataSet)
    bestFeatureAxis = -1
    # 保存最大的信息增益，此时分类后的信息熵最小
    maxInfoGain = 0.0
    for i in range(splitedCount):
        featureValueList = [temp[i] for temp in dataSet]
        uniqueFeatureValues = set(featureValueList)
        # 计算按照此分类效果的平均信息熵
        averageEntropy = 0.0
        for value in uniqueFeatureValues:
            splitData = getSplitDataSet(dataSet, i, value)
            p = float(len(splitData)) / dataSetLength
            # 取平均
            averageEntropy += p * calculateShannonEntropy(splitData)
        # 分类后的信息增益， averageEntropy越小越好
        infoGain = baseEntropy - averageEntropy
        if infoGain > maxInfoGain:  # 找到最大的信息增益
            maxInfoGain = infoGain
            bestFeatureAxis = i
    return bestFeatureAxis


def getSplitDataSet(dataSet, axis, value):
    """
    将哪一轴axis的数据作为分类类型，label为vale的数据作为一组数据
    :param dataSet:
    :param axis:
    :param value:
    :return:
    """
    splitDataSet = []
    for rowdata in dataSet:
        if rowdata[axis] == value:
            temp = rowdata[:]
            del (temp[axis])  # 去除已经分类的axis的值
            splitDataSet.append(temp)

    return splitDataSet


# ----------- 生成决策树 -----------


def majorityCnt(classList):
    """
    所有特征全部分类完，如果还有不同类型的数据在同一组，采用多数表决的方法
    决定该组数据该属于哪一组分类
    :param classList: 分类数据的label类别list
    :return:
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1),
                              reverse=True)

    return sortedClassCount[0][0]  # 返回类别最多的类别


def createDecisionTree(dataSet, labels):
    """
    递归创建决策树 （其中的yes/no针对本案例）
    :param dataSet: 原始输入数据数据中的元素满足[1, 0, 1, ... 'yes/no']
    :param labels: 所有的特征标签列表，例如：'color' 'size' ...的字符串
    :return:
    """
    # 所有的dataSet的类别信息，即yes/no
    classList = [temp[-1] for temp in dataSet]
    # 递归返回（或者说生成叶子节点的条件）的条件 #
    # 1、类别完全相同则停止继续划分数据集
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 类别相同，则直接返回该类别
    # 2、遍历完所有特征时返回出现次数最多的类别
    if len(dataSet[0]) == 1:  # 所有的特征都分类完，只剩下最后的类别
        return majorityCnt(classList)

    # 递归划分过程 #
    # 最好的划分所对应的列数axis
    bestFeatureAxis = chooseBestFeatureAxisToSplit(dataSet)
    # 最好的划分所对应的列数axis所对应的label，如按照'color'划分
    # 该label需要在决策树的节点显示，所以需要保存
    bestFeatureLabel = labels[bestFeatureAxis]
    # 删除已经划分的特征，对剩下的特征继续递归划分
    del (labels[bestFeatureAxis])
    # 创建决策树
    designTree = {bestFeatureLabel: {}}
    bestFeatureValues = [temp[bestFeatureAxis] for temp in dataSet]
    # 该特征下所有可能出现的值
    uniqueFeatureValues = set(bestFeatureValues)

    # 对每个值进行递归划分
    for value in uniqueFeatureValues:
        # 由于python函数传递的是引用，在del时会修改数据，所以此处需要保存labels副本
        subLabels = labels[:]
        # 按照此value分类的数据集
        splitDataSet = getSplitDataSet(dataSet, bestFeatureAxis, value)
        designTree[bestFeatureLabel][value] = createDecisionTree(splitDataSet, subLabels)

    return designTree


# ----------- 测试决策树分类器 -----------


def decisionTreeClassfy(decisionTree, featureLabels, inputTest):
    """
    测试决策树分类器，遍历决策树，比较测试数据和决策树上的数值，递归执行知道叶子节点，
    则该叶子节点所属类型即为待特使数据的类型
    :param decisionTree: 一直数据训练生成的决策树
    :param featureLabels: 决策树对应的特征标签列表
    :param inputTest: 输入的待分类测试数据
    :return:
    """
    bestFeatureLabel = decisionTree.keys()[0]  # 获取决策树的根节点
    # print 'bestFeatureLabel:', bestFeatureLabel
    # print 'featureLabels:', featureLabels
    bestFeatureAxis = featureLabels.index(bestFeatureLabel)  # 根节点feature所在labels的下标
    secondDict = decisionTree[bestFeatureLabel]  # 子树
    classLabel = None
    for key in secondDict.keys():  # 该bestFeatureLabel所有特征值
        if inputTest[bestFeatureAxis] == key:  # 如果测试的数据的bestFeatureLabel的特征值与某一个key相等
            if type(secondDict[key]).__name__ == 'dict':  # 递归调用
                classLabel = decisionTreeClassfy(secondDict[key], featureLabels, inputTest)
            else:  # 遍历到叶子节点
                classLabel = secondDict[key]

    return classLabel


# ----------- 存储学习过的决策树 -----------


def storeDecisionTree(decisionTree, fileName):
    """
    将决策树序列化到文件中，避免每次与测试都要学习数据创建决策树
    :param decisionTree:
    :param fileName:
    :return:
    """
    savefile = open(fileName, 'w')
    pickle.dump(decisionTree, savefile)
    savefile.close()


def getDecisionTreeFromFile(fileName):
    """
    从文件中反序列化得到决策树
    :param fileName:
    :return:
    """
    savefile = open(fileName, 'r')
    return pickle.load(savefile)
