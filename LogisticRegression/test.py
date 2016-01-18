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


def plotTest():
    featureDatas, classTypes = GradientRegression.loadDataSet()
    weights = GradientRegression.getBestRegressionWeightsByGradientAscent(featureDatas, classTypes)
    print weights
    GradientRegression.plotBestRegressionLine(featureDatas, classTypes, weights)


if __name__ == '__main__':
    # weightsTest()
    plotTest()
