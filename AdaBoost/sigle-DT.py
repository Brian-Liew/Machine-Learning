import numpy as np 
import matplotlib.pyplot as plt

def loadData():
	dataMat=np.matrix([[1. , 2.1],
		[1.5, 1.6],
		[1.3, 1. ],
		[1. , 1. ],
		[2. , 1. ]])
	classLabels=[1.0,1.0,-1.0,-1.0,1.0]
	return dataMat,classLabels

def showDataSet(dataMat, labelMat):

	data_plus = []                                  #正样本
	data_minus = []                                 #负样本
	for i in range(len(dataMat)):
		if labelMat[i] > 0:
			data_plus.append(dataMat[i])
		else:
			data_minus.append(dataMat[i])
	data_plus_np = np.array(data_plus)											 #转换为numpy矩阵
	data_minus_np = np.array(data_minus)										 #转换为numpy矩阵
	plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])		#正样本散点图
	plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1]) 	#负样本散点图
	plt.show()

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
	retArray=np.ones((np.shape(dataMatrix)[0],1))
	if threshIneq=='lt':
		retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
	else:
		retArray[dataMatrix[:,dimen]>threshVal]=-1.0
	return retArray

def buildStump(dataArr,classLabels,D):
	dataMatrix=np.mat(dataArr)
	labelMat=np.mat(classLabels).T 
	m,n =np.shape(dataMatrix)
	numSteps=10.0
	bestStump={}
	bestClasEst=np.mat(np.zeros((m,1)))
	minError=float('inf')
	for i in range(n):
		rangeMin=dataMatrix[:,i].min()
		rangeMax=dataMatrix[:,i].max()
		stepSize=(rangeMax-rangeMin)/numSteps
		for j in  range(-1,int(numSteps)+1):
			for inequal in ['lt','gt']:
				threshVal=(rangeMin+float(j)*stepSize)
				predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)
				errArr=np.mat(np.ones((m,1)))
				errArr[predictedVals==labelMat]=0
				weightedError=D.T*errArr
				print("split:dim %d, thresh %.2f,thresh ineuqal: %s,the weighted error is %.3f" % (i,threshVal,inequal,weightedError))
				if weightedError<minError:
					minError=weightedError
					bestClasEst=predictedVals.copy()
					bestStump['dim']=i
					bestStump['thresh']=threshVal
					bestStump['ineq']=inequal
	return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
	weakClassArr=[]
	m =np.shape(dataArr)[0]
	D=np.mat(np.ones((m,1))/m)
	aggClassEst=np.mat(np.zeros((m,1)))
	for i in range(numIt):
		bestStump,error,classEst=buildStump(dataArr,classLabels,D)
		#对于每一个样本的权值的赋予
		alpha=float(0.5*np.log((1.0-error)/max(error,1e-16)))
		bestStump['alpha']=alpha
		weakClassArr.append(bestStump)
		expon=np.multiply(-1*alpha*np.mat(classLabels).T,classEst)
		D=np.multiply(D,np.exp(expon))
		D=D/D.sum()
		aggClassEst+=alpha*classEst
		aggErrors=np.multiply(np.sign(aggClassEst)!=np.mat(classLabels).T,np.ones((m,1)))
		errRate=aggErrors.sum()/m 
		if errRate==0.0:
			break
	return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):
	dataMatrix=np.mat(datToClass)
	m=np.shape(dataMatrix)[0]
	aggClassEst=np.mat(np.zeros((m,1)))
	for i in range(len(classifierArr)):
		classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])

		aggClassEst+=classifierArr[i]['alpha']*classEst
		print(aggClassEst)
	return np.sign(aggClassEst)




dataArr,classLabels=loadData()
"""
D=np.mat(np.ones((5,1))/5)
bestStump,minError,bestClasEst=buildStump(dataArr,classLabels,D)
print('bestStump:\n',bestStump)
print('minError:\n',minError)
print('bestClasEst:\n',bestClasEst)
showDataSet(dataArr,classLabels)
"""
weakClassArr,aggClassEst=adaBoostTrainDS(dataArr,classLabels)
print(adaClassify([[0,0],[5,5]],weakClassArr))




