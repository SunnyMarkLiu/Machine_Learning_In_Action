#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""

import NavieBayesian as bayes


def createWordSetTest():
    wordsList, classTypes = bayes.loadDataSet()
    print wordsList
    wordsetList = bayes.createVocabularyList(wordsList)
    print wordsetList
    return wordsetList


def trainNavieBayesianTest():
    wordsList, classTypes = bayes.loadDataSet()
    vocaList = bayes.createVocabularyList(wordsList)
    # 将feature对应的标记为0,1
    trainVocabularyMattrix = []
    for words in wordsList:
        trainVocabularyMattrix.append(bayes.checkSignedFeatureList(vocaList, words))

    # print np.array(trainVocabularyMattrix)
    p_WiBasedOnClass0, p_WiBasedOnClass1, pAbusive = bayes.trainNavieBayesian(trainVocabularyMattrix, classTypes)
    print p_WiBasedOnClass0, '\n'
    print p_WiBasedOnClass1
    print pAbusive


def classifyNavieBayesianTest():
    wordsList, classTypes = bayes.loadDataSet()
    inputTestWords = ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please']
    result = bayes.classifyNavieBayesian(wordsList, classTypes, inputTestWords)
    print inputTestWords, ':', result
    inputTestWords2 = ['love', 'stupid']
    result2 = bayes.classifyNavieBayesian(wordsList, classTypes, inputTestWords2)
    print inputTestWords2, ':', result2

if __name__ == '__main__':
    # vocaList = createWordSetTest()
    # signedFeatureList = bayes.checkWordSetInInputWordSet(vocaList, ['my'])
    # print signedFeatureList
    # trainNavieBayesianTest()
    classifyNavieBayesianTest()
