import numpy as np 
import operator 
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN
def img2vector(filename):
	returnVect=np.zeros((1,1024))
	fr=open(filename)
	for i in range(32):
		linestr=fr.readline()
		for j in range(32):
			returnVect[0,32*i+j]=int(linestr[j])
	return returnVect

def handwritingClassTest():
	hwLabels=[]
	trainingFileList=listdir('trainingDigits')
	m=len(trainingFileList)
	trainingMat=np.zeros((m,1024))
	for i in range(m):
		filenameStr=trainingFileList[i]
		classNumber=int(filenameStr.split('_')[0])
		hwLabels.append(classNumber)
		trainingMat[i,:]=img2vector('trainingDigits/%s' %(filenameStr))
	clf=KNN(n_neighbors=3,algorithm='auto')
	clf.fit(trainingMat,hwLabels)
	testFilelist=listdir('testDigits')
	errorCount=0.0
	mTest=len(testFilelist)
	for i in range(mTest):
		filenameStr=testFilelist[i]
		classNumber=int(filenameStr.split('_')[0])
		vectorUnderTest=img2vector('testDigits/%s' %(filenameStr))
		classifierResult=clf.predict(vectorUnderTest)
		print("分类结果是%d\t真实结果是%d" %(classifierResult,classNumber))
		if(classifierResult!=classNumber):
			errorCount+=1.0
	print("总共错了%d个数据\n错误率是%f%%" %(errorCount,errorCount/mTest*100))

handwritingClassTest()