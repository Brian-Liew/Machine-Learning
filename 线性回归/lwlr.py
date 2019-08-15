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

def plotlwlr():
	xArr,yArr=loadDataSet('ex0.txt')
	yHat_1=lwlrTest(xArr,xArr,yArr,1.0)
	yHat_2=lwlrTest(xArr,xArr,yArr,0.01)
	yHat_3=lwlrTest(xArr,xArr,yArr,0.003)
	xMat=np.mat(xArr)
	yMat=np.mat(yArr)
	srtInd=xMat[:,1].argsort(0)
	xSort=xMat[srtInd][:,0,:]
	fig,axs=plt.subplots(nrows=3,ncols=1,sharex=False,sharey=False,figsize=(10,8))
	axs[0].plot(xSort[:,1],yHat_1[srtInd],c='red')
	axs[1].plot(xSort[:,1],yHat_2[srtInd],c='red')
	axs[2].plot(xSort[:,1],yHat_3[srtInd],c='red')
	axs[0].scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0],s=20,c='blue')
	axs[1].scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0],s=20,c='blue')
	axs[2].scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0],s=20,c='blue')
	axs0_title_text = axs[0].set_title(u'k=1.0')
	axs1_title_text = axs[1].set_title(u'k=0.01')
	axs2_title_text = axs[2].set_title(u'k=0.003')
	plt.xlabel('X')
	plt.show()

plotlwlr()