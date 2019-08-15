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

def regularize(xMat,yMat):
	inxMat=xMat.copy()
	inyMat=yMat.copy()
	yMean=np.mean(yMat,0)
	inyMat=inyMat-yMean
	xMean=np.mean(xMat,0)
	xVar=np.var(xMat,0)
	inxMat=(inxMat-xMean)/xVar
	return inxMat,inyMat

def rssError(yArr,yHatArr):
	return ((yArr-yHatArr)**2).sum()

def stageWise(xArr,yArr,eps=0.01,numIt=100):
	xMat=np.mat(xArr)
	yMat=np.mat(yArr).T
	xMat,yMat=regularize(xMat,yMat)
	m,n=np.shape(xMat)
	returnMat =np.zeros((numIt,n))
	ws=np.zeros((n,1))
	wsTest=ws.copy()
	wsMax=ws.copy()
	for i in range(numIt):
		lowestError=float('inf')
		for j in range(n):
			for sign in [-1,1]:
				wsTest=ws.copy()
				wsTest[j]+=eps*sign
				yTest=xMat*wsTest
				rssE=rssError(yMat.A,yTest.A)
				if rssE<lowestError:
					lowestError=rssE
					wsMax=wsTest
		ws=wsMax.copy()
		returnMat[i,:]=ws.T 
	return returnMat
def plot():
	abX,abY=loadDataSet('abalone.txt')
	w=stageWise(abX,abY,0.005,1000)
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.plot(w)
	plt.show()
plot()