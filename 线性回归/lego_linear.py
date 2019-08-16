import numpy as np 
from bs4 import BeautifulSoup
import random

def scrapePage(retX,retY,inFile,yr,numPce,origPrc):
	with open(inFile,encoding='utf-8') as f:
		html=f.read()
	soup =BeautifulSoup(html)
	i=1
	currentRow=soup.find_all('table',r="%d" %i)
	while(len(currentRow)!=0):
		currentRow=soup.find_all('table',r="%d" %i)
		title =currentRow[0].find_all('a')[1].text
		lwrTitle=title.lower()
		if(lwrTitle.find('new')>-1) or (lwrTitle.find('nisb')>-1):
			newFlag=1.0
		else:
			newFlag=0.0
		soldUnicde=currentRow[0].find_all('td')[3].find_all('span')
		if len(soldUnicde)==0:
			print("商品 #%d 没有出售" %i)
		else:
			soldPrice=currentRow[0].find_all('td')[4]
			priceStr=soldPrice.text
			priceStr=priceStr.replace('$','')
			priceStr=priceStr.replace(',','')
			if len(soldPrice)>1:
				priceStr=priceStr.replace('Free shipping','')
			sellingPrice=float(priceStr)
			if sellingPrice>origPrc*0.5:
				print("%d\t%d\t%d\t%f\t%f" %(yr,numPce,newFlag,origPrc,sellingPrice))
				retX.append([yr,numPce,newFlag,origPrc])
				retY.append(sellingPrice)
		i+=1
		currentRow=soup.find_all('table',r="%d" %i)


def setDataCollect(retX,retY):
	scrapePage(retX,retY,'./lego/lego8288.html',2006,800,49.99)
	scrapePage(retX,retY,'./lego/lego10030.html',2002,3096,269.99)
	scrapePage(retX,retY,'./lego/lego10179.html',2007,5195,499.99)
	scrapePage(retX,retY,'./lego/lego10181.html',2007,3428,199.99)
	scrapePage(retX,retY,'./lego/lego10189.html',2008,5922,299.99)
	scrapePage(retX,retY,'./lego/lego10196.html',2009,3263,249.99)

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

def crossValidation(xArr,yArr,numVal=10):
	m=len(yArr)
	indexList=list(range(m))
	errorMat=np.zeros((numVal,30))
	for i in range(numVal):
		trainX=[]
		trainY=[]
		testX=[]
		testY=[]
		random.shuffle(indexList)
		for j in range(m):
			if j<m*0.9:
				trainX.append(xArr[indexList[j]])
				trainY.append(yArr[indexList[j]])
			else:
				testX.append(xArr[indexList[j]])
				testY.append(yArr[indexList[j]])
		wMat=rigeTest(trainX,trainY)
		for k in range(30):
			matTestX=np.mat(testX)
			matTrainX=np.mat(trainX)
			meanTrain=np.mean(matTrainX,0)
			varTrain=np.var(matTrainX,0)
			matTestX=(matTestX-meanTrain)/varTrain
			yEst=matTestX*np.mat(wMat[k,:]).T+np.mean(trainY)
			errorMat[i,k]=rssError(yEst.T.A,np.array(testY))
	meanErrors=np.mean(errorMat,0)
	minMean=float(min(meanErrors))
	bestWeights=wMat[np.nonzero(meanErrors==minMean)]
	xMat=np.mat(xArr)
	yMat=np.mat(yArr).T
	meanX=np.mean(xMat,0)
	varX=np.var(xMat,0)
	unReg=bestWeights/varX
	print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % ((-1 * np.sum(np.multiply(meanX,unReg)) + np.mean(yMat)), unReg[0,0], unReg[0,1], unReg[0,2], unReg[0,3]))

lgX=[]
lgY=[]
setDataCollect(lgX,lgY)
crossValidation(lgX,lgY)
print(rigeTest(lgX,lgY))