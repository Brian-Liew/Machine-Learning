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

def standRegres(xArr,yArr):
	xMat=np.mat(xArr)
	yMat=np.mat(yArr).T 
	XTX=xMat.T*xMat
	if np.linalg.det(XTX)==0.0:
		print("矩阵奇异")
	w=XTX.I*(xMat.T*yMat)
	return w

def plotRegress():
	xArr,yArr=loadDataSet('ex0.txt')
	w=standRegres(xArr,yArr)
	xMat=np.mat(xArr)
	yMat=np.mat(yArr)
	xCopy=xMat.copy()
	xCopy.sort(0)
	yHat=xCopy*w
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.plot(xCopy[:,1],yHat,c='red')
	ax.scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0],s=20,c='blue')
	plt.title('DataSet')
	plt.xlabel('X')
	plt.show()

plotRegress()
