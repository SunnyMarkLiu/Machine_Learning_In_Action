#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""

from regression import *
import numpy as np
import matplotlib.pyplot as plt

datasArr, valuessArr = loadDataSet('datasets/ex0.txt')
# 利用标准回归计算系数W
ws = standardRegression(datasArr, valuessArr)

# 绘制原始数据
xMat = np.matrix(datasArr)
valueMat = np.matrix(valuessArr)
plt.figure(figsize=(10, 10), facecolor="white")
axes = plt.subplot(111)
plt.scatter(xMat[:, 1].flatten().A[0], valueMat.T.flatten().A[0])

# 绘制回归的曲线
xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy * ws
# 绘制预测的曲线
plt.plot(xCopy[:, 1], yHat)

plt.show()
