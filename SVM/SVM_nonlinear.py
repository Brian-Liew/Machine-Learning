import matplotlib.pyplot as plt 
import numpy as np 
import random

class optStruct:

	def __init__(self,dataMatIn,classLabels,C,toler,kTup):
		self.X=dataMatIn
		self.labelMat=classLabels
		self.C=C
		self.tol=toler
		self.m=np.shape(dataMatIn)[0]
		self.alphas=np.mat(np.zeros((self.m,1)))
		self.b=0
		self.eCache=np.mat(np.zeros((self.m,2)))
		self.K=np.mat(np.zeros((self.m,self.m)))
		for i in range(self.m):
			self.K[:,i]=kernelTrans(self.X,self.X[i,:],kTup)

def kernelTrans(X,A,kTup):
	m,n=np.shape(X)
	K=np.mat(np.zeros((m,1)))
	if kTup[0]=='lin':
		K=X*A.T
	elif kTup[0]=='rbf':
		for j in range(m):
			deltaRow=X[j,:]-A
			K[j]=deltaRow*deltaRow.T
		K = np.exp(K/(-1*kTup[1]**2))
	else:
		raise NameError('核函数无法识别')
	return K

def loadDataSet(filename):
	dataMat=[]
	lableMat=[]
	fr=open(filename)
	for line in fr.readlines():
		lineArr=line.strip().split('\t')
		dataMat.append([float(lineArr[0]),float(lineArr[1])])
		lableMat.append(float(lineArr[2]))
	return dataMat,lableMat


def calcEk(oS,k):
	fXk=float(np.multiply(oS.alphas,oS.labelMat).T*(oS.K[:,k])+oS.b)
	Ek=fXk-float(oS.labelMat[k])
	return Ek

def selectJrand(i,m):
	j=i
	while(j==i) :
		j=int(random.uniform(0,m))
	return j

def selectJ(i,oS,Ei):
	maxK=-1
	maxDeltaE=0
	Ej=0
	oS.eCache[i]=[1,Ei]
	validEcachelist=np.nonzero(oS.eCache[:,0].A)[0]
	if(len(validEcachelist))>1:
		for k in validEcachelist:
			if k==i :
				continue
			Ek=calcEk(oS,k)
			deltaE=abs(Ei-Ek)
			if(deltaE>maxDeltaE):
				maxK=k
				maxDeltaE=deltaE
				Ej=Ek
		return maxK,Ej
	else:
		j=selectJrand(i,oS.m)
		Ej=calcEk(oS,j)
	return j,Ej

def updateEk(oS,k):
	Ek=calcEk(oS,k)
	oS.eCache[k]=[1,Ek]

def clipAlpha(aj,H,L):
	if aj>H:
		aj=H
	if L>aj:
		aj=L
	return aj

def innerL(i,oS):
	Ei=calcEk(oS,i)
	#优化alpha
	if((oS.labelMat[i]*Ei<-oS.tol) and (oS.alphas[i]<oS.C)) or ((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i]>0)):
		#使用内循环
		j,Ej=selectJ(i,oS,Ei)
		#保存
		#alphaIold=oS.alphas[i].copy()
		#alphaJold=oS.alphas[j].copy()
		alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
		#计算上下界
		if (oS.labelMat[i] != oS.labelMat[j]):
			L = max(0, oS.alphas[j] - oS.alphas[i])
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
		else:
			L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
			H = min(oS.C, oS.alphas[j] + oS.alphas[i])
		if L == H:
			print("L==H")
			return 0
		#步骤三：计算eta
		eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
		if eta>=0:
			print("eta>=0")
			return 0
		oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
		#修建
		oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
		#更新误差
		updateEk(oS,j)
		if(abs(oS.alphas[j]-alphaJold)<0.00001):
			print("alpha_j变化太小")
			return 0
		oS.alphas[i]+=oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
		updateEk(oS,i)
		#更新b1，b2
		b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i]-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
		b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
		#更新b
		if (0<oS.alphas[i]) and (oS.C>oS.alphas[i]):
			oS.b=b1
		elif (0<oS.alphas[j]) and (oS.C>oS.alphas[j]):
			oS.b=b2
		else:
			oS.b=(b1+b2)/2.0
		return 1
	else:
		return 0

def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
	oS=optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler,kTup)
	iter=0
	entireSet=True
	alphaPairsChanged=0
	while(iter<maxIter) and ((alphaPairsChanged>0) or (entireSet)):
		alphaPairsChanged=0
		if entireSet:
			for i in range(oS.m):
				alphaPairsChanged+=innerL(i,oS)
				print("全样本遍历：第%d次迭代 样本：%d，alpha优化次数：%d" %(iter,i,alphaPairsChanged))
			iter+=1
		else:
			nonBoundIs = np.nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
			for i in nonBoundIs:
				alphaPairsChanged+=innerL(i,oS)
				print("非边界遍历：第%d次迭代 样本：%d，alpha优化次数：%d" % (iter,i,alphaPairsChanged))
			iter +=1
		if entireSet:
			entireSet=False
		elif(alphaPairsChanged==0):
			entireSet=True
		print("迭代次数：%d" % iter)
	return oS.b,oS.alphas

def testRbf(k1=1.3):
	dataArr,labelArr=loadDataSet('testSetRBF.txt')
	b,alphas=smoP(dataArr,labelArr,200,0.0001,100,('rbf',k1))
	dataMat = np.mat(dataArr)
	labelMat=np.mat(labelArr).transpose()
	svInd = np.nonzero(alphas.A>0)[0]
	sVs=dataMat[svInd]
	labelSV=labelMat[svInd]
	print("支持向量个数：%d" % np.shape(sVs)[0])
	m,n=np.shape(dataMat)
	errorCount=0
	for i in range(m):
		kernelEval =kernelTrans(sVs,dataMat[i,:],('rbf',k1))
		predict=kernelEval.T*np.multiply(labelSV,alphas[svInd])+b
		if np.sign(predict)!=np.sign(labelArr[i]):
			errorCount+=1
	print("训练集错误率： %.2f%%" %((float(errorCount)/m)*100))
	dataArr,labelArr=loadDataSet('testSetRBF2.txt')
	errorCount=0
	dataMat=np.mat(dataArr)
	labelMat=np.mat(labelArr)
	m,n=np.shape(dataMat)
	for i in range(m):
		kernelEval=kernelTrans(sVs,dataMat[i,:],('rbf',k1))
		predict=kernelEval.T*np.multiply(labelSV,alphas[svInd])+b
		if np.sign(predict)!=np.sign(labelArr[i]):
			errorCount+=1
	print("训练集错误率： %.2f%%" %((float(errorCount)/m)*100))
	showClassifer(dataArr,labelArr)

def showClassifer(dataMat,classLabels):
	data_plus=[]
	data_minus=[]
	for i in range(len(dataMat)):
		if classLabels[i]>0:
			data_plus.append(dataMat[i])
		else:
			data_minus.append(dataMat[i])
	data_plus_np=np.array(data_plus)
	data_minus_np=np.array(data_minus)
	plt.scatter(np.transpose(data_plus_np)[0],np.transpose(data_plus_np)[1],s=30,alpha=0.7)
	plt.scatter(np.transpose(data_minus_np)[0],np.transpose(data_minus_np)[1],s=30,alpha=0.7)
	plt.show()

testRbf()
