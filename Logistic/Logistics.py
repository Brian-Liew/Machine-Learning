import numpy as np 
import matplotlib.pyplot as plt 
"""
加载数据
把数据按一行一行都进去
然后返回数据矩阵以及相应的标签矩阵
"""
def loadDataSet():
	dataMat=[]
	labelMat=[]
	fr=open('testSet.txt')
	for line in fr.readlines() :
		lineArr = line.strip().split()
		dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	fr.close()
	return dataMat,labelMat

"""
sigmoid 函数：
返回sigmoid函数的值
"""
def sigmoid(inx):
	return 1.0/(1+np.exp(-inx))


def gradAscent(dataMatIn,classLabels):
	dataMatrix = np.mat(dataMatIn)
	labelMat = np.mat(classLabels).transpose()
	m,n =np.shape(dataMatrix)
	alpha =0.001
	maxCycles = 500
	weights=np.ones((n,1))
	for k in range(maxCycles):
		h=sigmoid(dataMatrix*weights)
		error=labelMat-h
		weights=weights+alpha*dataMatrix.transpose()*error
	return weights.getA()

dataMat,labelMat=loadDataSet()
print(gradAscent(dataMat,labelMat))

def plotFit(weights):
	dataMat,labelMat = loadDataSet()
	dataArr=np.array(dataMat)
	n=np.shape(dataMat)[0]
	xcord1=[];ycord1=[];
	xcord2=[];ycord2=[];
	for i in range(n):
		if int(labelMat[i])==1:
			xcord1.append(dataArr[i,1])
			ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1])
			ycord2.append(dataArr[i,2])
	fig = plt.figure()
	ax=fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)
	ax.scatter(xcord2,ycord2,s=20,c='green',alpha=.5)
	x=np.arange(-3.0,3.0,0.1)
	y=(-weights[0]-weights[1]*x)/weights[2]
	ax.plot(x,y)
	plt.title('Logistic')
	plt.xlabel('X1'); plt.ylabel('X2')
	plt.show() 

weights=gradAscent(dataMat,labelMat)
plotFit(weights)