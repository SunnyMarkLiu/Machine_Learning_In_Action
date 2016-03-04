#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""

import adaboost
import numpy as np

# 训练算法
dataMatrix, classLabels = adaboost.loadSimpData()
bestDecisionStumps = adaboost.adaboostTrainDecisionStump(dataMatrix, classLabels, 20)
print bestDecisionStumps

print "-------测试算法-------"
testDatas = [[0, 0], [5, 0]]
weightedForecastClasses, confidence = \
    adaboost.adaboostClassify(testDatas, bestDecisionStumps)

print "预测的结果及对应的分类把握："
print np.sign(weightedForecastClasses).T
print confidence.T
