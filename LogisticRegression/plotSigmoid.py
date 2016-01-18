#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
"""

import matplotlib.pyplot as plt
import numpy as np


def plotSigmoidTest():
    """
    绘制Sigmoid函数
    :return:
    """
    figure = plt.figure(figsize=(10, 10), facecolor="white")
    figure.clear()

    pltaxes = plt.subplot(111)
    num = np.linspace(-40, 40, 256, endpoint=True)

    y = 1.0 / (1 + np.exp(-1 * num))  # 计算Sigmoid函数 y(x) = 1 / (1 + exp(-z))
    pltaxes.plot(num, y, color="blue", linewidth=2.0, linestyle="-")
    plt.xlim(-40, 40)
    plt.savefig('plotSigmoidFunction.jpg')
    plt.show()


if __name__ == '__main__':
    plotSigmoidTest()
