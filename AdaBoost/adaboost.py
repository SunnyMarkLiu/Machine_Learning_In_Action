#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""
import numpy as np


def loadSimpData():
    """
    加载基本测试数据
    :return:
    """
    dataMatrix = np.matrix([[1., 2.1],
                            [2., 1.1],
                            [1.3, 1.],
                            [1., 1.],
                            [2., 1.]])
    classLabels = [1, 1, -1, -1, 1]
    return dataMatrix, classLabels


def simpleStumpClassify(dataMatrix, dimen, threshValue, sepOperator):
    """
    单层决策树分类函数
    :param dataMatrix: 输入的样本数据，不包括类别
    :param dimen: 从哪一个特征入手
    :param threshValue: 该特征分类的阈值
    :param sepOperator: 分类的操作符，即大于(greater_than)、小于(lower_than)阈值的情况
    :return: 返回预测的类别列向量
    """
    forecastClasses = np.ones((np.shape(dataMatrix)[0], 1))
    if sepOperator == 'lower_than':
        forecastClasses[dataMatrix[:, dimen] <= threshValue] = -1
    else:
        forecastClasses[dataMatrix[:, dimen] > threshValue] = -1

    return forecastClasses


def buildDecisionStump(trainDataMatrix, trainClasses, D):
    """
    通过训练数据创建简单决策树
    :param trainDataMatrix: 训练数据集
    :param trainClasses: 训练数据集的标签
    :param D: 训练数据集各个样本的权重矩阵
    :return: 返回创建的简单决策树所需的信息
    """
    # 保证数据为矩阵
    trainDataMatrix = np.matrix(trainDataMatrix)
    trainClasses = np.matrix(trainClasses).T    # 转置，便与计算
    D = np.matrix(D)

    m, n = np.shape(trainDataMatrix)
    stepsNum = 10  # 在此特征上迭代10次
    # 保存最佳分类的加权错误
    minWeightedError = np.inf   # 初始化设置加权分类错误率为正无穷,type为float
    bestDecisionStump = {}  # 保存最佳决策树
    bestPredictValue = np.matrix(np.zeros((m, 1)))   # 保存最佳分类结果
    # 对数据集的所有样本的特征进行遍历，以便寻找最佳分类的特征
    for diem in range(n):
        # 计算训练数据集在该特征的最大值和最小值的差别，以及每次的步长
        valueMin = trainDataMatrix[:, diem].min()
        valueMax = trainDataMatrix[:, diem].max()
        stepSize = (valueMax - valueMin) / stepsNum
        # 对该特征的min-max之间，以stepSize步进
        for j in range(-1, stepsNum + 1):
            # 由于此简单决策树分类器是二类分类器，所以遍历大于、小于作为分类分界，
            # 如果类别多于两种，则需要修改
            threshValue = valueMin + float(j) * stepSize
            for sepOperator in ['lower_than', 'greater_than']:
                # 计算决策树在该特征分类的阈值
                # 按照此分类标准（第几个特征diem，阈值threshValue，操作符sepOperator）
                # 分类的结果
                predictValues = simpleStumpClassify(trainDataMatrix, diem,
                                                    threshValue, sepOperator)
                # 用于标记分类错误的样本
                errArr = np.matrix(np.zeros((m, 1)))
                errArr[predictValues != trainClasses] = 1
                # 结合D的加权分类错误
                # D本身为列向量
                weightedError = float(D.T * errArr)    # 矩阵相乘计算内积

                # print '分类：从第 %d 个特征入手，阈值为 %.4f ， 操作符为 %s，分类的加权错误率为 %.4f' \
                #       % (diem, threshValue, sepOperator, weightedError)

                if weightedError < minWeightedError:
                    minWeightedError = float(weightedError)
                    # 保存最佳的决策树
                    bestDecisionStump['diem'] = diem
                    bestDecisionStump['threshValue'] = threshValue
                    bestDecisionStump['sepOperator'] = sepOperator
                    bestPredictValue = predictValues.copy()

    return bestDecisionStump, minWeightedError, bestPredictValue


def adaboostTrainDecisionStump(trainDataMatrix, trainClasses, iteratorCount=40):
    """
    基于单层决策树，对adaBoost算法进行训练
    :param trainDataMatrix: 训练数据集
    :param trainClasses: 训练数据集的标签
    :param iteratorCount: 训练迭代次数
    :return:
    """
    trainClasses = np.matrix(trainClasses)
    # 保存迭代训练获得的决策树
    bestDecisionStumps = []
    m = np.shape(trainDataMatrix)[0]
    # 训练数据集样本的权重初始化相等
    D = np.matrix(np.ones((m, 1)) / m)
    # 保存加权的最终预测结果
    finalPredictClass = np.matrix(np.zeros((m, 1)))
    for i in range(0, iteratorCount):
        print '——————第 %d 轮训练——————' % i
        print '权重D：', D.T
        bestDecisionStump, minWeightedError, bestPredictValue = \
            buildDecisionStump(trainDataMatrix, trainClasses, D)
        # 计算当前分类结果的错误率，作为样本的权重alpha

        alpha = 0.5 * np.log((1-minWeightedError) / max(minWeightedError, 1e-16))
        # 保存当前决策树分类结果的权重
        bestDecisionStump['alpha'] = alpha
        bestDecisionStumps.append(bestDecisionStump)

        print "本轮预测结果：", bestPredictValue.T
        # 根据前一轮预测获得的alpha结果权重更新样本的权重向量D
        # 前一轮预测结果正确的样本，减小其权重；预测结果错误的样本，增加其权重
        # 计算公式：
        # 前一轮预测结果正确：
        #   Di+1 = ( Di * exp(-alpha) ) / sum(Di)
        # 前一轮预测结果错误：
        #   Di+1 = ( Di * exp(alpha) ) / sum(Di)
        print "alpha:", alpha
        expon = np.multiply(-1*alpha*trainClasses.T, bestPredictValue)
        D = np.multiply(D, np.exp(expon)) / D.sum()

        # 加权预测结果
        finalPredictClass += alpha * bestPredictValue
        print "加权预测结果:", finalPredictClass.T
        # 加权后的预测结果错误的数目
        weightedErrors = np.multiply(np.sign(finalPredictClass) != trainClasses.T,
                                     np.ones((m, 1)))
        # 计算加权后的错误率
        errorRate = weightedErrors.sum() / m
        print "加权后此次训练的错误率:", errorRate
        if errorRate == 0.0:
            break

    return bestDecisionStumps


def adaboostClassify(testDataMatrix, bestDecisionStumps):
    """
    adaboost算法的分类函数
    :param testDataMatrix: 测试的数据集
    :param bestDecisionStumps: 训练adaboost算法获得的多个决策树策略
    :return:
    """
    testDataMatrix = np.matrix(testDataMatrix)
    m = np.shape(testDataMatrix)[0]
    weightedForecastClasses = np.matrix(np.zeros((m, 1)))
    for i in range(len(bestDecisionStumps)):    # 用每个决策树算法测试数据
        forecastClasses = simpleStumpClassify(testDataMatrix,
                                              bestDecisionStumps[i]['diem'],
                                              bestDecisionStumps[i]['threshValue'],
                                              bestDecisionStumps[i]['sepOperator'])
        # 计算加权后的类别
        weightedForecastClasses += bestDecisionStumps[i]['alpha'] * forecastClasses
        print weightedForecastClasses

    confidence = calConfidence(weightedForecastClasses)
    return weightedForecastClasses, confidence


def calConfidence(weightedForecastClasses):
    """
    由预测的权重类别，计算分类的把握
    :param weightedForecastClasses: 预测的权重类别
    :return:
    """
    m = np.shape(weightedForecastClasses)[0]
    # confidence = np.matrix(np.zeros(np.shape((m, 1))))
    confidence = 1 / (1+np.exp(-1*weightedForecastClasses))
    for i in range(0, len(confidence)):
        if confidence[i] < 0.5:
            confidence[i] = 1 - confidence[i]
    return confidence
