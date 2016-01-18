#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

import kNN
import matplotlib.pyplot as plt

group, labels = kNN.createDataSet()
input_data = [1.1, 1.2]
result = kNN.knn_classfy(input_data, group, labels, 3)
print '数据', input_data, '属于：', result

# create a new figure and set it's width, height
plt.figure(figsize=(10, 10))
plt.subplot(111)
plt.scatter(group[:, 0], group[:, 1], c='#ef6790', s=20)
for label, x, y in zip(labels, group[:, 0], group[:, 1]):
    plt.annotate(label,
                 xy=(x, y),  # 所要注释的坐标
                 xytext=(-10, 0),  # 注释文字中心偏离（x,y）坐标的位移
                 textcoords='offset points',  # important！设置text坐标是相对于（x,y)的偏移
                 ha='right',  # 设置水平对齐horizontalalignment
                 va='bottom')  # 设置垂直对齐verticalalignment

# 绘制预测的结果：
plt.scatter(input_data[0], input_data[1], c='#ff0000', s=80)
plt.annotate(result,
             xy=(input_data[0], input_data[1]),
             xytext=(-10, 0),
             textcoords='offset points',
             ha='right',
             va='bottom')

plt.show()
