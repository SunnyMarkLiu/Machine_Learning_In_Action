#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_
"""
Created on Nov 22, 2010
@author: Peter
"""

from numpy import *
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import svmMLiA

xcord0 = []  # 类别0的x坐标
ycord0 = []  # 类别0的y坐标
xcord1 = []  # 类别1的x坐标
ycord1 = []  # 类别1的y坐标
markers = []
colors = []

dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
C = 0.6
b, alphas = svmMLiA.smoSimple(dataArr, labelArr, C, 0.001, 40)
b = float(b)
# 支持向量的坐标
xSupportVectors = []
ySupportVectors = []
alphaSupportVectors = []  # 支持向量对应的非零alpha
labelSupportVectors = []
featureDatasSupportVectors = []  # 支持向量的原始数据
for i in range(0, alphas.__len__()):
    if alphas[i] > 0:
        print i, ":", alphas[i]
        alphaSupportVectors.append(float(alphas[i]))
        labelSupportVectors.append(int(labelArr[i]))
        xSupportVectors.append(dataArr[i][0])
        ySupportVectors.append(dataArr[i][1])
        featureDatasSupportVectors.append(dataArr[i])

print "支持向量的坐标："
print xSupportVectors, ySupportVectors
print "支持向量的alpha："
print alphaSupportVectors
print "支持向量的类别："
print labelSupportVectors
print "支持向量的特征数据："
print featureDatasSupportVectors
print "b:"
print b

for i in range(0, len(dataArr)):
    xPt = float(dataArr[i][0])  # X坐标
    yPt = float(dataArr[i][1])  # Y坐标
    label = int(labelArr[i])  # 所属类别
    if label == -1:
        xcord0.append(xPt)
        ycord0.append(yPt)
    else:
        xcord1.append(xPt)
        ycord1.append(yPt)

# 计算向量W，参考公式
W = svmMLiA.calcWs(alphas, dataArr, labelArr)
# W = zeros((1, len(dataArr[0])))
# for i in range(0, len(alphaSupportVectors)):
#     Wi = array([alphaSupportVectors[i] * labelSupportVectors[i] * featureData for featureData in
#                 featureDatasSupportVectors[i]])
#     W = W + Wi

print "超平面的法向量W："
print W

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xcord0, ycord0, marker='s', s=90)
ax.scatter(xcord1, ycord1, marker='o', s=50, c='red')
plt.title('Support Vectors Circled')
# 此处手动指定支持向量的坐标
for i in range(0, len(xSupportVectors)):
    circle = Circle((xSupportVectors[i], ySupportVectors[i]), 0.3, facecolor='none', edgecolor=(0, 0.8, 0.8),
                    linewidth=3, alpha=0.5)
    ax.add_patch(circle)
# plt.plot([2.3,8.5], [-6,6]) #seperating hyperplane

# w0,w1,b确定最优超平面WTX+b=0,
# 由于此案例是二维的数据，w0X + w1Y +b =0为hyperplane
# 解的y=(-w0X-b)/w1
w0 = float(W[0])
w1 = float(W[1])
x = arange(-2.0, 12.0, 0.1)
y = (-w0 * x - b) / w1
ax.plot(x, y)
ax.axis([-2, 12, -8, 6])
plt.show()
