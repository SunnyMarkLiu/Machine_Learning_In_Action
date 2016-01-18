#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
绘制决策树
@author: MarkLiu
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc')
decisionNodeArgs = dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5)
leafNodeArgs = dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5)
arrowArgs = dict(arrowstyle='<-', connectionstyle='arc3,rad=0', color='red')


def plotNode(plotaxes, nodeText, centerPointer, parentPointer, nodeTypeArgs):
    """
    绘制节点
    :param plotaxes:
    :param nodeText:
    :param centerPointer: (x,y)所要注释显示的坐标
    :param parentPointer: 父节点的坐标
    :param nodeTypeArgs:
    :return:
    """
    plotaxes.annotate(nodeText,
                      xy=parentPointer,  # 所要注释的坐标
                      xycoords='axes fraction',
                      xytext=centerPointer,  # 注释文字中心偏离（x,y）坐标的位移
                      textcoords='axes fraction',  # important！设置text坐标是相对于（x,y)的偏移
                      ha='center',  # 设置水平对齐horizontalalignment
                      va='center',  # 设置垂直对齐verticalalignment
                      bbox=nodeTypeArgs,
                      arrowprops=arrowArgs,
                      fontproperties=zhfont)


def plotMidText(plotaxes, cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    plotaxes.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(plotaxes, myTree, parentPointer, nodeTxt):  # if the first key tells you what feat was split on
    numLeafs = getLeafNumber(myTree)  # this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]  # the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(plotaxes, cntrPt, parentPointer, nodeTxt)
    plotNode(plotaxes, firstStr, cntrPt, parentPointer, decisionNodeArgs)
    secondDict = myTree[firstStr]
    plotTree.yOff -= 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(plotaxes, secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff += 1.0 / plotTree.totalW
            plotNode(plotaxes, secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNodeArgs)
            plotMidText(plotaxes, (plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff += 1.0 / plotTree.totalD


# if you do get a dictonary you know it's a tree, and the first element will be another dict

def plotDecisionTress(inTree):
    """
    绘制决策树
    :param inTree:
    :return:
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    plotaxes = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getLeafNumber(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(plotaxes, inTree, (0.5, 1.0), '')
    plt.show()


def getLeafNumber(decisionTree):
    """
    获取叶子节点的数目，以便确定绘制的x轴的坐标范围
    :param decisionTree:
    :return:
    """
    leafNumber = 0
    features = decisionTree.keys()[0]
    subDict = decisionTree[features]
    for key in subDict.keys():
        if type(subDict[key]).__name__ == 'dict':
            leafNumber += getLeafNumber(subDict[key])
        else:
            leafNumber += 1

    return leafNumber


def getTreeDepth(decisionTree):
    """
    获取决策树的深度，以便确定y轴坐标范围
    :param decisionTree:
    :return:
    """
    maxTreeDepth = 0
    features = decisionTree.keys()[0]
    subDict = decisionTree[features]
    for key in subDict.keys():
        if type(subDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(subDict[key])
        else:
            thisDepth = 2  # 根节点 + 叶子节点 最低深度为2

        if thisDepth > maxTreeDepth:
            maxTreeDepth = thisDepth

    return maxTreeDepth
