#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
案例：基于概率论的朴素贝叶斯算法实现过滤垃圾邮件
@Author: MarkLiu
"""


def loadEmailText():
    """
    加载email文本数据，包括垃圾邮件和非垃圾邮件
    :return:
    """
    for i in range(1, 26):
        emailContent = open('email/harm/%d.txt' % i).read()
        print emailContent


def filterSpamEmail():
    """
    过滤垃圾邮件
    :return:
    """
    loadEmailText()

if __name__ == '__main__':
    filterSpamEmail()
