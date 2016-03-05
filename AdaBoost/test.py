#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
利用adaboost算法预测患有马疝病的马是否会死亡
@Author: MarkLiu
"""
import adaboost
import numpy as np


def loadDataSet(fileName):
    """
    可根据数据集的特征的个数自适应加载数据
    :param fileName:
    :return:
    """
    frTrain = open(fileName)
    datasArr = []
    labelsArr = []
    for line in frTrain.readlines():
        linedatas = line.strip().split('\t')
        floatdatas = []
        for i in range(len(linedatas)):
            floatdatas.append(float(linedatas[i]))
        datasArr.append(floatdatas[:-1])
        labelsArr.append(int(floatdatas[-1]))

    frTrain.close()
    return datasArr, labelsArr


if __name__ == '__main__':
    trainDatas, trainLabels = loadDataSet('dataset/trainingDatas.txt')
    print np.shape(trainDatas)
    print trainDatas
    print np.shape(trainLabels)
    print trainLabels

    # 训练算法获得多个简单决策树分类器
    bestDecisionStumps = adaboost.adaboostTrainDecisionStump(trainDatas, trainLabels, 40)
    testDatas, testLabels = loadDataSet('dataset/testDatas.txt')
    # 返回预测结果
    weightedForecastClasses, confidence = \
        adaboost.adaboostClassify(testDatas, bestDecisionStumps)
    # 统计分类错误
    errorArr = np.matrix(np.zeros((len(testLabels), 1)))
    errorArr[np.sign(weightedForecastClasses) != np.matrix(testLabels).T] = 1
    print "errorArr:"
    print errorArr.T
    print "分类的结果："
    print "预测的类别：", np.sign(weightedForecastClasses).T
    print "测试数据共 %d 个" % len(testLabels)
    print "错误分类共 %d 个" % errorArr.sum()
    print "分类的错误率为：", 1.0 * errorArr.sum() / len(testLabels)
