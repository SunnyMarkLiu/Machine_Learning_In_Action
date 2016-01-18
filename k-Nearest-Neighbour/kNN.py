#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

import numpy as np
import operator  # 包括abs，add等操作方法


def createDataSet():
    """
    创建数据集
    """
    group = np.array([[1.0, 1.1],
                      [1.1, 1.3],
                      [1.3, 1.0],
                      [0.9, 1.0],
                      [0.89, 1.12],
                      [1.0, 1.0],
                      [0.1, 0.14],
                      [0.4, 0.12],
                      [0.6, 0.34],
                      [0.2, 0.22],
                      [0.12, 0.1],
                      [0.31, 0.32],
                      [0, 0.1]])
    labels = ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B']
    return group, labels


def autoNormalize(data_matrix):
    """
    将数据进行归一化处理
    newvalue = (oldvalue - min) / (max - min)
    :param data_matrix: 待归一化的矩阵
    :return:
    """
    # 1. 获取矩阵每一列对应的最大值和最小值
    min_values = data_matrix.min(axis=0)  # axis=0 表示列
    max_values = data_matrix.max(axis=0)
    # 2. 计算每一列的取值范围
    ranges = max_values - min_values
    # 初始化一个和原矩阵大小相等的矩阵用于保存归一化的数据
    normalize_datas = np.zeros(shape=np.shape(data_matrix))
    # 原数据的行数
    line_count = data_matrix.shape[0]
    # 3. 利用公式救出归一化的数据newvalue = (oldvalue - min) / (max - min)
    # 将min_values一行的矩阵扩展成和data_matrix系统的形式，并和data_matrix对应位置数据做差
    normalize_datas = data_matrix - np.tile(min_values, 1)
    normalize_datas = normalize_datas / np.tile(ranges, 1)
    # 4. 返回归一化的矩阵数据
    return normalize_datas


def file2matrix(filename):
    """
    将文件中的数据转换为矩阵
    :param filename:
    :return:
    """
    f = open(filename)
    all_lines_list = f.readlines()
    number_of_lines = len(all_lines_list)
    # 初始化待返回的矩阵 number_of_lines x 3
    return_matrix = np.zeros((number_of_lines, 3))
    return_labels = []
    index = 0
    for line in all_lines_list:
        line = line.strip()
        line_data_list = line.split('\t')
        return_matrix[index, :] = line_data_list[0:3]
        return_labels.append(int(line_data_list[-1]))
        index += 1
    return return_matrix, return_labels


def knn_classfy(inputdata, sample_data_set, labels, k):
    """
    kNN算法的分类
    :param inputdata: 待分类的输入数据
    :param sample_data_set: 待训练的样本数据集
    :param labels: 结果类型为标签
    :param k: k-近邻算法的k值
    :return: 返回该分类的输入数据锁对应的结果类型
    """
    # 样本数据的数目，对应为样本矩阵的行数
    sample_data_count = sample_data_set.shape[0]
    # 将inputdata数据复制扩展成sample_data_count×1
    formated_inputdata = np.tile(inputdata, (sample_data_count, 1))
    # 利用距离公式计算矩阵中各个点对应之间的距离
    # formated_inputdata - sample_data_set 为两矩阵对应位置的值相减
    # matrix.sum(axis=1)) 计算矩阵每一行的和
    distances = (((formated_inputdata - sample_data_set) ** 2).sum(axis=1)) ** 0.5
    # distances排序后的数据对应在distances的list中的位置
    sorted_data_indexs = distances.argsort()

    label_count_dict = {}  # 保存label及其出现的次数{'A':1, }
    for i in range(k):  # 取出前k个距离最近的
        label = labels[sorted_data_indexs[i]]  # 获取对应的label
        label_count_dict[label] = label_count_dict.get(label, 0) + 1
    sorted_label_count_list = sorted(label_count_dict.iteritems(),
                                     key=operator.itemgetter(1),
                                     reverse=True)
    return sorted_label_count_list[0][0]
