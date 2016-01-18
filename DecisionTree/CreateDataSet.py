#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_


def createDataSet():
    """
    获取测试数据集
    :return:
    """
    dataSet = [[1, 1, 1, 1, 'yes'],
               [1, 0, 1, 1, 'no'],
               [2, 2, 1, 1, 'yes'],
               [2, 2, 0, 1, 'no'],
               [2, 2, 0, 1, 'no'],
               [0, 1, 1, 1, 'noaksjdh'],
               [0, 1, 1, 0, 'yes'],
               [0, 1, 0, 1, 'yes']]
    labels = ['no surfacing', 'flippers', 'another', 'head']
    return dataSet, labels
