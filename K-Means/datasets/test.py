#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""
import numpy as np

arr = np.loadtxt('testSet.txt', delimiter='\t')
print arr
print np.shape(arr)
print type(arr[0][0])
np.savetxt('savearray.txt', arr, delimiter='\t')
