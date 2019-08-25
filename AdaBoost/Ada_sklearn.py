import numpy as np 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def loadData(filename):
	numFeat=len((open(filename).readline().split('\t')))
	dataMat=[]
	labelMat=[]
	fr=open(filename)
	for line in fr.readlines():
		lineArr=[]
		curLine=line.strip().split('\t')
		for i in range(numFeat-1):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat,labelMat
dataArr,classLabels=loadData('horseColicTraining.txt')
testArr,testLabelArr=loadData('horseColicTest.txt')
bdt=AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),algorithm="SAMME",n_estimators=10)
bdt.fit(dataArr,classLabels)
predictions=bdt.predict(dataArr)
errArr=np.mat(np.ones((len(dataArr),1)))
print('训练集错误率：%.3f%%' %float(errArr[predictions!=classLabels].sum()/len(dataArr)*100))
predictions=bdt.predict(testArr)
errArr=np.mat(np.ones((len(testArr),1)))
print('测试集错误率：%.3f%%' %float(errArr[predictions!=testLabelArr].sum()/len(testArr)*100))