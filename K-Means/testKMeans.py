#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""

import numpy as np
import kMeans
import matplotlib.pyplot as plt

dataArr = kMeans.loadDataSet('datasets/testSet2.txt')
dataMat = np.matrix(dataArr)
k = 3
centroids, clusterAssment = kMeans.biKmeans(dataMat, k)
# centroids, clusterAssment = kMeans.kMeans(dataMat, k)

# 计算原始数据加上中心数据，将数据分离
m = np.shape(dataMat)[0]

# 分离出不同簇的x，y坐标
xPoint_0 = []
yPoint_0 = []
xPoint_1 = []
yPoint_1 = []
xPoint_2 = []
yPoint_2 = []
xPoint_3 = []
yPoint_3 = []
for i in range(m):
    if int(clusterAssment[i, 0]) == 0:
        xPoint_0.append(dataMat[i, 0])
        yPoint_0.append(dataMat[i, 1])
    if int(clusterAssment[i, 0]) == 1:
        xPoint_1.append(dataMat[i, 0])
        yPoint_1.append(dataMat[i, 1])
    if int(clusterAssment[i, 0]) == 2:
        xPoint_2.append(dataMat[i, 0])
        yPoint_2.append(dataMat[i, 1])
    if int(clusterAssment[i, 0]) == 3:
        xPoint_3.append(dataMat[i, 0])
        yPoint_3.append(dataMat[i, 1])  # 绘制聚类的数据

plt.figure(figsize=(10, 10), facecolor="white")
plt.subplot(211)
xPoints = dataMat[:, 0].flatten().A[0]
yPoints = dataMat[:, 1].flatten().A[0]
plt.scatter(xPoints, yPoints, s=160, c='blue', marker='o')
plt.subplot(212)
plt.scatter(xPoint_0, yPoint_0, s=160, c='blue', marker='o')
plt.scatter(xPoint_1, yPoint_1, s=160, c='red', marker='o')
plt.scatter(xPoint_2, yPoint_2, s=160, c='green', marker='o')
plt.scatter(xPoint_3, yPoint_3, s=160, c='black', marker='o')
plt.savefig("kMeans.png", dip=72)
plt.show()
