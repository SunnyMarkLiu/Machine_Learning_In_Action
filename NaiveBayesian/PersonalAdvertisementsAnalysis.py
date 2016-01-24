#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
使用朴素贝叶斯分类器从个人广告中获取区域倾向
@Author: MarkLiu
"""
import feedparser

result = feedparser.parse('http://beijing.craigslist.com.cn/stp/index.rss')
for i in range(len(result['entries'])):
    print result['entries'][i]['summary_detail']['value']
print type(result['entries'])
print len(result['entries'])
