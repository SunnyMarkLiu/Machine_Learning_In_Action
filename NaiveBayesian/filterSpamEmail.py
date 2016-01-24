#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
案例：基于概率论的朴素贝叶斯算法实现过滤垃圾邮件
@Author: MarkLiu
"""
import NavieBayesian as bayes
import random


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


def loadEmailText():
    """
    加载email文本数据，包括垃圾邮件和非垃圾邮件
    :return:
    """
    # 未作处理的原始的文档集合
    initialDocList = []
    # 文档类别列表
    classTypes = []

    for i in range(1, 26):
        textList = textParser(open('email/ham/%d.txt' % i).read())
        initialDocList.append(textList)
        classTypes.append(0)
        textList = textParser(open('email/spam/%d.txt' % i).read())
        initialDocList.append(textList)
        classTypes.append(1)

    return initialDocList, classTypes


def filterSpamEmail():
    """
    过滤垃圾邮件
    :return:
    """
    initialDocList, classTypes = loadEmailText()
    # 从initialDocList中随机创建10个待测试的文档
    testDocList = []
    # 待测试邮件的类型
    testDocClassList = []
    """
    注意此处随机选择10封email，添加到测试集合，同时将原有的数据集删除，
    这种随机选择数据的一部分作为训练集合，而剩余部分作为测试集合的过程称为
    留存交叉验证：hold-out cross validation
    """
    for i in range(10):
        randomIndex = int(random.uniform(0, len(initialDocList)))
        testDocClassList.append(classTypes[randomIndex])
        testDocList.append(initialDocList[randomIndex])
        del(initialDocList[randomIndex])
        del(classTypes[randomIndex])

    print 'testDocList:', len(testDocList)
    print 'initialDocList:', len(initialDocList)
    print 'testDocClassList:', len(testDocClassList)
    print 'classTypes:', len(classTypes)
    errorCount = 0
    for i in range(len(testDocList)):
        classType = bayes.classifyNavieBayesian(
                initialDocList, classTypes, testDocList[i])
        if classType != testDocClassList[i]:  # 预测的结果和实际的结果进行比较
            errorCount += 1

    print 'the error rate is :', float(errorCount) / len(testDocList)


if __name__ == '__main__':
    filterSpamEmail()
