import numpy as np 
import matplotlib.pyplot as plt 

def loadDataSet(filename):
	fr=open(filename)
	xArr=[]
	yArr=[]
	numFeat=len(open(filename).readline().split('\t'))-1
	for line in fr.readlines():
		lineArr=[]
		curline=line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curline[i]))
		xArr.append(lineArr)
		yArr.append(float(curline[-1]))
	return xArr,yArr

def lwlr(testPoint,xArr,yArr,k=1.0):
	xMat=np.mat(xArr)
	yMat=np.mat(yArr).T 
	m=np.shape(xMat)[0]
	weight=np.mat(np.eye((m)))
	for j in range(m):
		diffMat=testPoint-xMat[j,:]
		weight[j,j]=np.exp(diffMat*diffMat.T/(-2.0*k**2))
	xTx=xMat.T*weight*xMat
	if np.linalg.det(xTx)==0.0:
		print('矩阵奇异')
		return 
	w=xTx.I*(xMat.T*(weight*yMat))
	return testPoint*w
def lwlrTest(testArr,xArr,yArr,k=1.0):
	m=np.shape(testArr)[0]
	yHat=np.zeros(m)
	for i in range(m):
		yHat[i]=lwlr(testArr[i],xArr,yArr,k)
	return yHat

def standRegres(xArr,yArr):
	xMat=np.mat(xArr)
	yMat=np.mat(yArr).T 
	XTX=xMat.T*xMat
	if np.linalg.det(XTX)==0.0:
		print("矩阵奇异")
	w=XTX.I*(xMat.T*yMat)
	return w

def rssError(yArr,yHatArr):
	return ((yArr-yHatArr)**2).sum()

abX,abY=loadDataSet('abalone.txt')
yHat01=lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
yHat02=lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
yHat03=lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
print('训练集相同的情况：')
print('k=0.1,误差是：',rssError(abY[0:99],yHat01.T))
print('k=1,误差是：',rssError(abY[0:99],yHat02.T))
print('k=10,误差是：',rssError(abY[0:99],yHat03.T))
yHat01=lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
yHat02=lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
yHat03=lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)
print('训练集不同的情况：')
print('k=0.1,误差是：',rssError(abY[100:199],yHat01.T))
print('k=1,误差是：',rssError(abY[100:199],yHat02.T))
print('k=10,误差是：',rssError(abY[100:199],yHat03.T))
w=standRegres(abX[0:99],abY[0:99])
yHat=np.mat(abX[100:199])*w
print('简单的线性回归误差：',rssError(abY[100:199],yHat.T.A))
