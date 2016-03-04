#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""

import adaboost
import numpy as np

dataMatrix, classLabels = adaboost.loadSimpData()
# D = np.matrix(np.ones((5, 1)) / 5)
bestDecisionStumps = adaboost.adaboostTrainDecisionStump(dataMatrix, classLabels, 9)
print bestDecisionStumps
# bestDecisionStump, minWeightedError, bestPredictValue = \
#     adaboost.buildDecisionStump(dataMatrix, classLabels, D)
#
# print '---------------------'
# print classLabels
# print bestDecisionStump
# print bestPredictValue
# print minWeightedError
