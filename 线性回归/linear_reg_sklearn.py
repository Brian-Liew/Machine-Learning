import numpy as np 
from bs4 import BeautifulSoup
import random
from sklearn import linear_model

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

def usesklearn():
	reg=linear_model.Ridge(alpha=.5)
	lgX=[]
	lgY=[]
	setDataCollect(lgX,lgY)
	reg.fit(lgX,lgY)
	print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' %(reg.intercept_,reg.coef_[0],reg.coef_[1],reg.coef_[2],reg.coef_[3]))

usesklearn()