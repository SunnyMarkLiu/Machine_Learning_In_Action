#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""
import svmMLiA

dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
C = 0.6
b, alphas = svmMLiA.smoSimple(dataArr, labelArr, C, 0.001, 40)
print "SVM执行结束"
print "b:"
print b
print "alphas:"
print alphas[alphas > 0]

xPoints = []
yPoints = []
for i in range(0, alphas.__len__()):
    if alphas[i] > 0:
        print alphas[i]
        xPoints.append(dataArr[i][0])
        yPoints.append(dataArr[i][1])

print "xPoints:", xPoints
print "yPoints:", yPoints
