from time import sleep
import matplotlib.pyplot as plt 
import numpy as np 
import random
import types

def loadDataSet(filename):
	dataMat=[]
	lableMat=[]
	fr=open(filename)
	for line in fr.readlines():
		lineArr=line.strip().split('\t')
		dataMat.append([float(lineArr[0]),float(lineArr[1])])
		lableMat.append(float(lineArr[2]))
	return dataMat,lableMat

#随机选择alpha值
def selectJrand(i,m):
	j=i
	while(j==i) :
		j=int(random.uniform(0,m))
	return j

#修剪alpha
def clipAlpha(aj,H,L):
	if aj>H:
		aj=H
	if L>aj:
		aj=L
	return aj

#简化版smo算法：
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
	dataMatrix =np.mat(dataMatIn)
	lableMat=np.mat(classLabels).transpose()
	b=0
	m,n=np.shape(dataMatrix)
	alphas=np.mat(np.zeros((m,1)))
	iter_num=0
	while(iter_num<maxIter):
		alphaPairsChanged=0
		for i in range(m):
			fXi=float(np.multiply(alphas,lableMat).T*(dataMatrix*dataMatrix[i,:].T))+b
			Ei=fXi-float(lableMat[i])
			#优化alpha
			if((lableMat[i]*Ei< -toler) and (alphas[i]<C)) or ((lableMat[i]*Ei>toler) and (alphas[i]>0)):
				j=selectJrand(i,m)
				fXj=float(np.multiply(alphas,lableMat).T*(dataMatrix*dataMatrix[j,:].T))+b
				Ej=fXj-float(lableMat[j])
				alphaIold =alphas[i].copy()
				alphaJold=alphas[j].copy()
				#计算上下界
				if(lableMat[i]!=lableMat[j]):
					L=max(0,alphas[j]-alphas[i])
					H=min(C,C+alphas[j]-alphas[i])
				else:
					L=max(0,alphas[j]+alphas[i]-C)
					H=min(C,alphas[j]+alphas[i])
				if L==H:
					print("L==H")
					continue
				#步骤3：计算eta
				eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
				if eta>=0:
					print("eta>=0")
					continue
				#更新alpha_j
				alphas[j] -= lableMat[j]*(Ei-Ej)/eta
				alphas[j]=clipAlpha(alphas[j],H,L)
				if(abs(alphas[j]-alphaJold)<0.00001):
					print("alpha_j变化太小")
					continue
				alphas[i]+=lableMat[j]*lableMat[i]*(alphaJold-alphas[j])
				b1=b-Ei-lableMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-lableMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
				b2=b-Ej-lableMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-lableMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
				#更新b
				if(0<alphas[i]) and (C>alphas[i]):
					b=b1
				elif (0<alphas[j]) and (C>alphas[j]):
					b=b2 
				else:
					b=(b1+b2)/2.0
				alphaPairsChanged+=1
				print("第%d次迭代 样本：%d， alpha优化次数：%d" %(iter_num,i,alphaPairsChanged))
		if(alphaPairsChanged==0):
			iter_num+=1
		else:
			iter_num=0
		print("迭代次数： %d" % iter_num)
	return b,alphas

def showClassifer(dataMat,w,b):
	data_plus=[]
	data_minus=[]
	for i in range(len(dataMat)):
		if lableMat[i]>0:
			data_plus.append(dataMat[i])
		else:
			data_minus.append(dataMat[i])
	data_plus_np=np.array(data_plus)
	data_minus_np=np.array(data_minus)
	plt.scatter(np.transpose(data_plus_np)[0],np.transpose(data_plus_np)[1],s=30,alpha=0.7)
	plt.scatter(np.transpose(data_minus_np)[0],np.transpose(data_minus_np)[1],s=30,alpha=0.7)
	x1=max(dataMat)[0]
	x2=min(dataMat)[0]
	a1,a2=w
	b=float(b)
	a1=float(a1[0])
	a2=float(a2[0])
	y1=(-b-a1*x1)/a2
	y2= (-b-a1*x2)/a2
	plt.plot([x1,x2],[y1,y2])
	for i,alpha in enumerate(alphas):
		if alpha>0:
			x,y=dataMat[i]
			plt.scatter([x],[y],s=150,c='none',alpha=0.7,linewidth=1.5,edgecolor='red')
	plt.show()

def get_w(dataMat,lableMat,alphas):
	alphas,dataMat,lableMat=np.array(alphas),np.array(dataMat),np.array(lableMat)
	w=np.dot((np.tile(lableMat.reshape(1,-1).T,(1,2))*dataMat).T,alphas)
	return w.tolist()

dataMat,lableMat=loadDataSet('testSet.txt')
b,alphas=smoSimple(dataMat,lableMat,0.6,0.001,40)

w=get_w(dataMat,lableMat,alphas)

showClassifer(dataMat,w,b)