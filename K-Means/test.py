#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""

import numpy as np

data = np.matrix([[1, 2], [3, 4]])
print data
data1 = data + 1
print data1
data2 = data1 * 2
print data2
data3 = np.power(data2, 2)
print data3
data4 = np.sum(data3)
print data4
print '-------'
print np.random.rand(2, 2)
