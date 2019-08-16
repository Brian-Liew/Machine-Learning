import numpy as np 

def loadDataSet(filename):
	dataMat=[]
	fr = open(filename)
	for line in fr.readlines():
		curLine=line.strip().split('\t')
		fltLine=list(map(float,curLine))
		dataMat.append(fltLine)
	return dataMat

def binSplitDataSet(dataSet,feature,value):
	mat0=dataSet[np.nonzero(dataSet[:,feature]>value)[0],:]
	mat1=dataSet[np.nonzero(dataSet[:,feature]<=value)[0],:]
	return mat0,mat1

def regLeaf(dataSet):
	#生成叶节点(均值)
	return np.mean(dataSet[:,-1])

def regErr(dataSet):
	#误差估计函数(方差)
	return np.var(dataSet[:,-1])*np.shape(dataSet)[0]

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
	import types
	#tols允许的误差下降值，tolN切分的最少样本数
	tolS=ops[0]
	tolN=ops[1]
	#如果所有值相等，退出
	if len(set(dataSet[:,-1].T.tolist()[0]))==1:
		return None,leafType(dataSet)
	#默认最后一个特征为最佳切分，计算误差
	m,n=np.shape(dataSet)
	S=errType(dataSet)
	bestS=float('inf')
	bestIndex=0
	bestValue=0
	for featIndex in range(n-1):
		for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):
			mat0,mat1=binSplitDataSet(dataSet,featIndex,splitVal)
			if(np.shape(mat0)[0]<tolN) or (np.shape(mat1)[0]<tolN):
				continue
			newS=errType(mat0)+errType(mat1)
			if newS<bestS:
				bestIndex=featIndex
				bestValue=splitVal
				bestS=newS
	if(S-bestS)<tolS:
		return None,leafType(dataSet)
	mat0,mat1=binSplitDataSet(dataSet,bestIndex,bestValue)
	if(np.shape(mat0)[0]<tolN) or (np.shape(mat1)[0]<tolN):
		return None,leafType(dataSet)
	return bestIndex,bestValue

def creatTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
	feat,val=chooseBestSplit(dataSet,leafType,errType,ops)
	if feat==None:
		return val
	retTree={}
	retTree['spInd']=feat
	retTree['spVal']=val
	lSet,rSet=binSplitDataSet(dataSet,feat,val)
	retTree['left']=creatTree(lSet,leafType,errType,ops)
	retTree['right']=creatTree(rSet,leafType,errType,ops)
	return retTree

def isTree(obj):
	import types
	return (type(obj).__name__=='dict')

def getMean(tree):
	if isTree(tree['right']):
		tree['right']=getMean(tree['right'])
		tree['left']=getMean(tree['left'])
	return (tree['left']+tree['right'])/2.0

def prune(tree,testData):
	if np.shape(testData)[0]==0:
		return getMean(tree)
	if (isTree(tree['right']) or isTree(tree['left'])):
		lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
	if isTree(tree['left']):
		tree['left']=prune(tree['left'],lSet)
	if isTree(tree['right']):
		tree['right']=prune(tree['right'],rSet)
	if not isTree(tree['left']) and not isTree(tree['right']):
		lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
		errorNomerge=np.sum(np.power(lSet[:,-1]-tree['left'],2))+np.sum(np.power(rSet[:,-1]-tree['right'],2))
		treeMean=(tree['left']+tree['right'])/2.0
		errMerge=np.sum(np.power(testData[:,-1]-treeMean,2))
		if errMerge<errorNomerge:
			return treeMean
		else:
			return tree
	else:
		return tree


train_filename='ex2.txt'
train_data=loadDataSet(train_filename)
train_Mat=np.mat(train_data)
tree=creatTree(train_Mat)
print('剪枝前：')
print(tree)
test_filename='ex2test.txt'
test_Data=loadDataSet(test_filename)
test_Mat=np.mat(test_Data)
print('\n 剪枝后：')
print(prune(tree,test_Mat))