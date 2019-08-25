import numpy as np 
import matplotlib.pyplot as plt

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

def plotRoc(predicStrengths,classLabels):
	cur=(1.0,1.0)
	ySum=0.0
	numPosClas=np.sum(np.array(classLabels)==1.0)
	yStep=1/float(numPosClas)
	xStep=1/float(len(classLabels)-numPosClas)
	sortedIndicties=predicStrengths.argsort()
	fig=plt.figure()
	fig.clf()
	ax=plt.subplot(111)
	for index in sortedIndicties.tolist()[0]:
		if classLabels[index]==1.0:
			delX=0
			delY=yStep
		else:
			delX=xStep
			delY=0
			ySum+=cur[1]
		ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY])
		cur=(cur[0]-delX,cur[1]-delY)
	plt.title('ROC')
	plt.xlabel('Fall')
	plt.ylabel('Recall')
	ax.axis([0,1,0,1])
	print('Auc面积：',ySum*xStep)
	plt.show()

dataArr,LabelArr=loadData('horseColicTraining2.txt')
weakClassArr,aggClassEst=adaBoostTrainDS(dataArr,LabelArr,50)
plotRoc(aggClassEst.T,LabelArr)