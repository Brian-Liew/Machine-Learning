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

def rigeRegress(xMat,yMat,lam=0.2):
	xTx=xMat.T*xMat
	denom=xTx+np.eye(np.shape(xMat)[1])*lam
	if np.linalg.det(denom)==0.0:
		print("矩阵为奇异矩阵，不能转置")
		return 
	w=denom.I*(xMat.T*yMat)
	return w

def rigeTest(xArr,yArr):
	xMat=np.mat(xArr)
	yMat=np.mat(yArr).T
	ymean=np.mean(yMat,axis=0)
	yMat=yMat-ymean
	xMeans=np.mean(xMat,axis=0)
	xVar=np.var(xMat,axis=0)
	xMat=(xMat-xMeans)/xVar
	numTestPts=30
	wMat=np.zeros((numTestPts,np.shape(xMat)[1]))
	for i in range(numTestPts):
		ws=rigeRegress(xMat,yMat,np.exp(i-10))
		wMat[i,:]=ws.T
	return wMat

def plot():
	abX,abY=loadDataSet('abalone.txt')
	w=rigeTest(abX,abY)
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.plot(w)
	plt.show()
plot()