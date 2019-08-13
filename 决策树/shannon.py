from math import log
import operator

def calcShannonEnt(dataSet):
	numEntire=len(dataSet)
	labelCounts={}
	for featVec in dataSet :
		currentLabel =featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel]=0
		labelCounts[currentLabel]+=1
	shannonEnt=0.0
	for key in labelCounts:
		prob  =float(labelCounts[key])/numEntire
		shannonEnt-=prob*log(prob,2)
	return shannonEnt

def createDataSet():
	dataSet=[[0, 0, 0, 0, 'no'],
			[0, 0, 0, 1, 'no'],
			[0, 1, 0, 1, 'yes'],
			[0, 1, 1, 0, 'yes'],
			[0, 0, 0, 0, 'no'],
			[1, 0, 0, 0, 'no'],
			[1, 0, 0, 1, 'no'],
			[1, 1, 1, 1, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[2, 0, 1, 2, 'yes'],
			[2, 0, 1, 1, 'yes'],
			[2, 1, 0, 1, 'yes'],
			[2, 1, 0, 2, 'yes'],
			[2, 0, 0, 0, 'no']]
	labels=['年龄','有工作','有自己的房子','信贷情况']
	return dataSet,labels

def splitDataSet(dataSet,axis,value):
	returnDateSet=[]
	for featVec in dataSet:
		if featVec[axis]==value:
			reducedFeatVec=featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			returnDateSet.append(reducedFeatVec)
	return returnDateSet

def chooseBest(dataSet):
	numFeatures=len(dataSet[0])-1
	baseEntropy=calcShannonEnt(dataSet)
	bestInforGain =0.0
	bestFeature=-1
	for i in range(numFeatures):
		featlist=[example[i] for example in dataSet]
		uniqueVals=set(featlist)
		newEntropy=0.0
		for value in uniqueVals:
			subDataSet=splitDataSet(dataSet,i,value)
			prob=len(subDataSet)/float(len(dataSet))
			newEntropy+=prob*calcShannonEnt(subDataSet)
		infoGain=baseEntropy-newEntropy
		if (infoGain>bestInforGain):
			bestInforGain=infoGain
			bestFeature=i 
		return bestFeature

def majorityCnt(classlist):
	classCount={}
	for vote in classlist:
		if vote not in classCount.keys():
			classCount[vote]=0
		classCount[vote]+=1
	sortClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	return sortClassCount[0][0]

def createTree(dataSet,labels,featLabels):
	classlist=[example[-1] for example in dataSet]
	if classlist.count(classlist[0])==len(classlist):
		return classlist[0]
	if len(dataSet[0])==1 or len(labels)==0:
		return majorityCnt(classlist)
	bestFeat=chooseBest(dataSet)
	bestFeatLabel=labels[bestFeat]
	featLabels.append(bestFeatLabel)
	mytree={bestFeatLabel:{}}
	del(labels[bestFeat])
	featValues=[example[bestFeat] for example in dataSet]
	uniqueVals=set(featValues)
	for value in uniqueVals:
		mytree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),labels,featLabels)
	return mytree

def classify(inputTree,featLabels,testVec):
	firstStr=next(iter(inputTree))
	secondDict=inputTree[firstStr]
	featIndex=featLabels.index(firstStr)
	for key in secondDict.keys():	
		if testVec[featIndex]==key:
			if type(secondDict[key]).__name__=='dict':
				classlabel=classify(secondDict[key],featLabels,testVec)
			else:
				classlabel=secondDict[key]
	return classlabel

dataSet,labels=createDataSet()
featLabels=[]
mytree=createTree(dataSet,labels,featLabels)
print(mytree)
testVec=[0,1]
result=classify(mytree,featLabels,testVec)
if(result=='yes'):
	print('放贷')
else:
	print('不放贷')



