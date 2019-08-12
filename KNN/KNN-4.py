import numpy as np 
import operator
"""
inX是测试集
dataSet是训练集
"""
def classif0(inX,dataSet,labels,k):
	#行数
	m=dataSet.shape[0]
	#扩展单行的测试集
	diffMat=np.tile(inX,(m,1))-dataSet
	sqMat=diffMat**2
	sqDisance=sqMat.sum(axis=1)
	distance=sqDisance*0.5
	sort_distance=distance.argsort()
	classCount={}
	for i in range(k):
		votelabel=labels[sort_distance[i]]
		classCount[votelabel]=classCount.get(votelabel,0)+1
	sortClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	return sortClassCount[0][0]

def file2matrix(filename):
	fr=open(filename)
	lines=fr.readlines()
	number=len(lines)
	returnMat =np.zeros((number,3))
	classLabelVector=[]
	index=0
	for line in lines:
		line=line.strip().split('\t')
		returnMat[index,:]=line[0:3]
		if line[-1]=='didntLike':
			classLabelVector.append(1)
		elif line[-1]=='smallDoses':
			classLabelVector.append(2)
		elif line[-1]=='largeDoses':
			classLabelVector.append(3)
		index+=1
	return returnMat,classLabelVector

def autoNorm(dataSet):
	minVal=dataSet.min(0)
	maxVal=dataSet.max(0)
	ranges=maxVal-minVal
	normDataSet=np.zeros(np.shape(dataSet))
	m=dataSet.shape[0]
	normDataSet=dataSet-np.tile(minVal,(m,1))
	normDataSet=normDataSet/np.tile(ranges,(m,1))
	return normDataSet,ranges,minVal

def datingClassTest():
	filename="datingTestSet.txt"
	datingDataMat,datingLabels=file2matrix(filename)
	hoRatio=0.10
	normMat,ranges,minVal=autoNorm(datingDataMat)
	m=normMat.shape[0]
	num_test=int(m*hoRatio)
	errorCount=0.0
	for i in range(num_test):
		classResult=classif0(normMat[i,:],normMat[num_test:m,:],datingLabels[num_test:m],4)
		print("分类结果：%d\t真是类别：%d" % (classResult,datingLabels[i]))
		if classResult!=datingLabels[i]:
			errorCount+=1.0
	print("错误率：%f%%" %(errorCount/float(num_test)*100))

datingClassTest()