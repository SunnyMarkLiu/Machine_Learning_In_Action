#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
使用朴素贝叶斯分类器从个人广告中获取区域倾向
@Author: MarkLiu
"""
import feedparser
import random
import NavieBayesian as bayes


def textParser(emailContent):
    """
    字符串解析成指定的格式
    :param emailContent:
    :return:
    """
    import re
    textList = re.split(re.compile('\W*'), emailContent)
    textList = [temp.lower() for temp in textList if len(temp) > 0]
    return textList


def loadRSSText(city0Rss, city1Rss):
    """
    加载RSS源数据，获取原始词汇列表
    :param city0Rss:
    :param city1Rss:
    :return:
    """
    # 未作处理的原始的文档集合
    initialDocList = []
    # 城市类别列表
    cityTypes = []
    # 所有出现的词汇
    fullText = []

    entries0 = city0Rss['entries']
    entries1 = city1Rss['entries']
    minLen = min(len(entries0), len(entries1))
    for i in range(minLen):
        textList = textParser(entries0[i]['summary'])
        initialDocList.append(textList)
        fullText.extend(textList)
        cityTypes.append(0)
        textList = textParser(entries1[i]['summary'])
        initialDocList.append(textList)
        fullText.extend(textList)
        cityTypes.append(1)

    return initialDocList, fullText, cityTypes


def calcFrequentWords(voclist, fullText):
    """
    去除高频词汇
    :param voclist:
    :param fullText:
    :return:
    """
    import operator
    freqDict = {}
    for token in voclist:
        freqDict[token] = fullText.count(token)

    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:(len(sortedFreq) / 10)]  # 取排序后的前1/10


def getCityTopWords(trainCity0Rss, trainCity1Rss):
    """
    获取城市中评论中最多的词汇
    :param trainCity1Rss:
    :param trainCity0Rss:
    """
    initialDocList, fullText, cityTypes = loadRSSText(trainCity0Rss, trainCity1Rss)
    vocaList = bayes.createVocabularyList(initialDocList)
    trainVocabularyMattrix = []
    # 将训练的文档集合针对vocaList进行标记
    for words in initialDocList:
        signedFeatureList = bayes.checkSignedFeatureList(vocaList, words)
        trainVocabularyMattrix.append(signedFeatureList)

    p_WiBasedOnClass0, p_WiBasedOnClass1, pAbusive = \
        bayes.trainNavieBayesian(trainVocabularyMattrix, cityTypes)

    topCity0Words = []
    topCity1Words = []
    for i in range(len(p_WiBasedOnClass0)):
        if p_WiBasedOnClass0[i] > -6.0:
            topCity0Words.append(vocaList[i])
        if p_WiBasedOnClass1[i] > -6.0:
            topCity1Words.append(vocaList[i])

    print '*******City0最常用20的词汇*********'
    for word in topCity0Words[:20]:
        print word
    print '*******City1最常用的词汇*********'
    for word in topCity1Words[:20]:
        print word


def localWordsTest(city0Rss, city1Rss):
    """
    测试根据输入的text分类城市的准确率
    :param city0Rss:
    :param city1Rss:
    过滤垃圾邮件
    :return:
    """
    initialDocList, fullText, cityTypes = loadRSSText(city0Rss, city1Rss)
    voclist = bayes.createVocabularyList(initialDocList)
    print '未删除高频词汇的词汇表长度：', len(voclist)
    # 出现频率最高的词汇，例如：I and 等辅助词
    deletedVoc = calcFrequentWords(voclist, fullText)
    # 去除词汇列表的高频词汇
    for word in deletedVoc:
        if word[0] in voclist:
            voclist.remove(word[0])
    print '删除后的词汇表长度：', len(voclist)

    # 从initialDocList中随机创建10个待测试的文档
    testDocList = []
    # 待测试邮件的类型
    testDocClassList = []
    """
    注意此处随机选择10个数据，添加到测试集合，同时将原有的数据集删除，
    这种随机选择数据的一部分作为训练集合，而剩余部分作为测试集合的过程称为
    留存交叉验证：hold-out cross validation
    """
    for i in range(10):
        randomIndex = int(random.uniform(0, len(initialDocList)))
        testDocClassList.append(cityTypes[randomIndex])
        testDocList.append(initialDocList[randomIndex])
        del (initialDocList[randomIndex])
        del (cityTypes[randomIndex])

    errorCount = 0
    for j in range(len(testDocList)):
        classType = bayes.classifyNavieBayesian2(
                voclist, initialDocList, cityTypes, testDocList[j])
        if classType != testDocClassList[j]:  # 预测的结果和实际的结果进行比较
            print '分类错误的信息：', testDocList[j], '\n属于', testDocClassList[j], \
                '错误分类成了：', classType
            errorCount += 1

    # 计算分类的误差
    errorRate = float(errorCount) / len(testDocList)
    print 'the error rate is :', errorRate
    return errorRate


if __name__ == '__main__':
    newyork = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    beijing = feedparser.parse('http://beijing.craigslist.com.cn/stp/index.rss')
    # errorSum = 0.0
    # for m in range(20):
    #     errorrate = localWordsTest(newyork, beijing)
    #     errorSum += errorrate
    #
    # print 'the average error rate is :', errorSum / 20.0
    getCityTopWords(newyork, beijing)
