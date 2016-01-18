#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

import matplotlib
import matplotlib.pyplot as plt
import kNN

reload(kNN)
matrix, labels = kNN.file2matrix('datingTestSet.txt')
print matrix
# print labels
zhfont = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc')

""" 比较好看的绘制方法 """

plt.figure(figsize=(8, 5), dpi=80)
axes = plt.subplot(111)
# 将三类数据分别取出来
# x轴代表飞行的里程数
# y轴代表玩视频游戏的百分比
type1_x = []
type1_y = []
type2_x = []
type2_y = []
type3_x = []
type3_y = []
for i in range(len(labels)):
    if labels[i] == 1:  # 不喜欢
        type1_x.append(matrix[i][0])
        type1_y.append(matrix[i][1])

    if labels[i] == 2:  # 魅力一般
        type2_x.append(matrix[i][0])
        type2_y.append(matrix[i][1])

    if labels[i] == 3:  # 极具魅力
        type3_x.append(matrix[i][0])
        type3_y.append(matrix[i][1])

type1 = axes.scatter(type1_x, type1_y, s=20, c='red')
type2 = axes.scatter(type2_x, type2_y, s=40, c='green')
type3 = axes.scatter(type3_x, type3_y, s=50, c='blue')
# plt.scatter(matrix[:, 0], matrix[:, 2], s=20 * numpy.array(labels),
#             c=50 * numpy.array(labels), marker='o',
#             label='test')
plt.xlabel(u'每年获取的飞行里程数', fontproperties=zhfont)
plt.ylabel(u'玩视频游戏所消耗的事件百分比', fontproperties=zhfont)
axes.legend((type1, type2, type3), (u'不喜欢', u'魅力一般', u'极具魅力'), loc=2, prop=zhfont)

plt.show()


def dating_test():
    """
    KNN分类器针对约会网站的测试
    :return:
    """
    # 测试数据占总数据的百分比
    test_ratio = 0.1
    # 归一化数据
    normalized_matrix = kNN.autoNormalize(matrix)
    total_number = normalized_matrix.shape[0]
    column = normalized_matrix.shape[1]
    # 测试的数据总个数
    test_number = int(total_number * test_ratio)
    # 记录分类错误的个数
    error_count = 0
    for j in range(test_number):
        """
            注意此处取出了第一列和第三列的数据
            由于前面的此时发现飞行的里程数和玩视频游戏分类的效果更明显
        """
        test_data = normalized_matrix[j, 0:2]  # => normalized_matrix[i]
        sample_data_set = normalized_matrix[test_number:total_number, 0:2]
        # KNN算法获得的分类结果
        classfy_result = kNN.knn_classfy(test_data, sample_data_set, labels[test_number:total_number], 6)
        # 实际的label结果
        real_result = labels[j]
        print '预测的结果：%d, 实际的结果：%d' % (classfy_result, real_result)
        if classfy_result != real_result:
            error_count += 1.0
    print '分类的错误率：%f' % (error_count / test_number)  # 最好结果0.040000


dating_test()
