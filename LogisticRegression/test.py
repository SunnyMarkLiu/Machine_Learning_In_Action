#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""
import GradientRegression


def weightsTest():
    featureDatas, classTypes = GradientRegression.loadDataSet()
    weights = GradientRegression.getBestRegressionWeightsByGradientAscent(featureDatas, classTypes)
    print weights


if __name__ == '__main__':
    weightsTest()
