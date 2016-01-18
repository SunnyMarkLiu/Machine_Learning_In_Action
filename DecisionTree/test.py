#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

import CreateDataSet
import DecisionTrees
import PlotDecisionTree


# dataSet, labels = CreateDataSet.createDataSet()
# featureLabels = labels[:]
# decisionTrees = DecisionTrees.createDecisionTree(dataSet, labels)
# PlotDecisionTree.plotDecisionTress(decisionTrees)
# inputTest = [0, 1, 1, 1]
# # DecisionTrees.storeDecisionTree(decisionTrees, 'DecisionTrees.txt')
# decisionTrees = DecisionTrees.getDecisionTreeFromFile('DecisionTrees.txt')
# print type(decisionTrees)
# classType = DecisionTrees.decisionTreeClassfy(decisionTrees, featureLabels, inputTest)
# print classType

# --------- glass分类测试 ---------#

def getGlassDataSet():
    classTypes = ['refractive_index', 'Sodium', 'Magnesium', 'Aluminum Silicon', ',Potassium', 'Calcium', 'Barium',
                  'Iron']
    glassfile = open('GlassData/glass.txt', 'r')
    lines = glassfile.readlines()
    count = len(lines)
    print count
    glassData = []
    for line in lines:
        line = line.strip()
        lineList = line.split(',')
        floatlist = []
        for i in range(1, len(lineList) - 1):
            floatlist.append(float(lineList[i]))
        floatlist.append(classTypes[int(lineList[-1])])
        glassData.append(floatlist)

    glassFeaturefile = open('GlassData/glass-feature.txt', 'r')
    line = glassFeaturefile.readline()
    line = line.strip()
    featureList = line.split(' ')
    glassfile.close()
    glassFeaturefile.close()
    return glassData, featureList


glassDataSet, featureLabels = getGlassDataSet()
# decisionTrees = DecisionTrees.createDecisionTree(glassDataSet, featureLabels)
# PlotDecisionTree.plotDecisionTress(decisionTrees)
# DecisionTrees.storeDecisionTree(decisionTrees, 'glassDecisionTrees.txt')

decisionTrees = DecisionTrees.getDecisionTreeFromFile('glassDecisionTrees.txt')
PlotDecisionTree.plotDecisionTress(decisionTrees)
inputTest = [1.52101, 3.64, 14.49, 1.10, 21.78, 1.06, 8.75, 0.00, 2.00]
classType = DecisionTrees.decisionTreeClassfy(decisionTrees, featureLabels, inputTest)
print inputTest, '所属类别：', classType

"""
   Type of glass: (class attribute)
      -- 1 building_windows_float_processed
      -- 2 building_windows_non_float_processed
      -- 3 vehicle_windows_float_processed
      -- 4 vehicle_windows_non_float_processed (none in this database)
      -- 5 containers
      -- 6 tableware
      -- 7 headlamps
"""
