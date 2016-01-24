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
    vocaList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classTypes = [0, 1, 0, 1, 0, 1]  # 1 代表侮辱性文字, 0 代表正常文字
    return vocaList, classTypes


def createWordSet(wordsList):
    """
    创建不重复的词汇列表，返回list。
    其中该词汇列表，可以作为判断文本片段的特征
    :param wordsList:
    :return:
    """
    wordSet = set()
    for words in wordsList:
        wordSet = wordSet | set(words)
    return list(wordSet)


def checkSignedFeatureList(vocabularyList, inputWords):
    """
    测试输入的文本向量中的文本是否在文档集合中有，并标记。
    返回文档集合标记的列表，可理解为特征值列表
    :type vocabularyList: list
    :param vocabularyList:
    :param inputWords:
    :return:
    """
    # 创建固定长度的list
    signedFeatureList = [0] * len(vocabularyList)
    for word in inputWords:
        if word in vocabularyList:  # 如果word在输入的文本向量中，则将对应的特征标记
            signedFeatureList[vocabularyList.index(word)] = 1

    return signedFeatureList


def trainNavieBayesian(trainVocabularyMattrix, trainClassTypes):
    """
    Navie Bayesian分类器的训练函数
    :param trainVocabularyMattrix: 训练文本向量的集合
    :param trainClassTypes: 训练词汇的类别
    :return:
    """
    # 训练文本向量的数目
    trainCount = len(trainVocabularyMattrix)
    # 文本向量的特征数目
    numFeature = len(trainVocabularyMattrix[0])
    # 存在侮辱性文字的文本向量占训练数据的的概率
    pAbusive = sum(trainClassTypes) / float(trainCount)
    """
    计算P(Wi|C1)和P(Wi|C0)
    由于在词汇表中存在多种词汇，计算概率时，使用Numpy数组快速计算。
    在for循环中，遍历训练集合trainVocabularyMattrix中的所有文档，一旦某个
    词语（侮辱性或正常）在某一文档中出现，则该词语对应的个数（class0FeatureNum或class1FeatureNum）
    就加1,同时文档出现的总词数也加1.
    注意：
    判断某个文档属于某个类别时，计算P(W|Ci)，W为向量
    即P(W0|C1)*P(W1|C1)*...*P(Wi|C1) 或者C0
    如果某个词汇没有出现则P(Wi|C1)为0,最终结果也为0.
    为避免这种情况，将词汇出现的次数初始化为1,分母初始化为2
    """
    # class0类别0对应的各个feature出现的数目
    class0FeatureNum = np.ones(numFeature)
    class1FeatureNum = np.ones(numFeature)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(trainCount):  # 对每一篇训练文档
        if trainClassTypes[i] == 1:  # 如果此文档是侮辱性
            class1FeatureNum += trainVocabularyMattrix[i]
            p1Denom += sum(trainVocabularyMattrix[i])
        else:
            class0FeatureNum += trainVocabularyMattrix[i]
            p0Denom += sum(trainVocabularyMattrix[i])

    # P(Wi|C0)：p0条件下词汇出现的条件概率,取log避免数据过小相乘出现下溢出
    p_WiBasedOnClass0 = np.log(class0FeatureNum / p0Denom)
    # P(Wi|C1)：p1条件下词汇出现的条件概率
    p_WiBasedOnClass1 = np.log(class1FeatureNum / p1Denom)

    return p_WiBasedOnClass0, p_WiBasedOnClass1, pAbusive


def classifyNavieBayesian(inputTestWords):
    """
    贝叶斯分类函数
    :type inputTestWords: list
    :param inputTestWords:
    :return:
    """
    wordsList, classTypes = loadDataSet()
    vocaList = createWordSet(wordsList)
    # 将feature对应的标记为0,1
    trainVocabularyMattrix = []
    for words in wordsList:
        signedFeatureList = checkSignedFeatureList(vocaList, words)
        trainVocabularyMattrix.append(signedFeatureList)

    p_WiBasedOnClass0, p_WiBasedOnClass1, pAbusive = trainNavieBayesian(trainVocabularyMattrix, classTypes)

    # 将inputTestWords文档字符串列表标记
    inputTestVec = checkSignedFeatureList(vocaList, inputTestWords)

    # 计算P(Ci|W)，W为向量。P(Ci|W)只需计算P(W|Ci)P(Ci)
    p1 = sum(inputTestVec * p_WiBasedOnClass1) + np.log(pAbusive)
    p0 = sum(inputTestVec * p_WiBasedOnClass0) + np.log(1 - pAbusive)

    if p1 > p0:
        return 1
    else:
        return 0
