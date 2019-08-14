import numpy as np
import random
import re 

def creatVocList(dataSet):
	vocaSet=set([])
	for document in dataSet:
		#不断取并集
		vocaSet=vocaSet | set(document)
	return list(vocaSet)

def set2Vec(vocablist,inputSet):
	returnVec=[0]*len(vocablist)
	for word in inputSet:
		if word in vocablist:
			returnVec[vocablist.index(word)]=1
		else:print("the word: %s is not in my vocablist!" %word)
	return returnVec


def trainNB0(trainMatrix,trainCategory):
	numTrainDocs=len(trainMatrix)
	numWords=len(trainMatrix[0])
	pAbusive=sum(trainCategory)/float(numTrainDocs)
	p0Num=np.ones(numWords)
	p1Num=np.ones(numWords)
	p0Denom=2.0
	p1Denom=2.0
	for i in range(numTrainDocs):
		if trainCategory[i]==1:
			p1Num+=trainMatrix[i]
			p1Denom+=sum(trainMatrix[i])
		else:
			p0Num+=trainMatrix[i]
			p0Denom+=sum(trainMatrix[i])
	p1Vect=np.log(p1Num/p1Denom)
	p0Vect=np.log(p0Num/p0Denom)
	return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
	p1=sum(vec2Classify*p1Vec)+np.log(pClass1)
	p2=sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
	if p1>p2:
		return 1
	else:
		return 0

def textParse(bigString):
	listOfTokens=re.split(r'\W+',bigString)
	return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
	doclist=[]
	classlist=[]
	fullText=[]
	for i in range(1,26):
		wordlist=textParse(open('email/spam/%d.txt' %i ,'r').read())
		doclist.append(wordlist)
		fullText.append(wordlist)
		classlist.append(1)
		wordlist=textParse(open('email/ham/%d.txt' %i ,'r').read())
		doclist.append(wordlist)
		fullText.append(wordlist)
		classlist.append(0)
	vocablist=creatVocList(doclist)
	trainingSet=list(range(50))
	testSet=[]
	for i in range(10):
		randIndex=int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMat=[]
	trainClass=[]
	for docIndex in trainingSet:
		trainMat.append(set2Vec(vocablist,doclist[docIndex]))
		trainClass.append(classlist[docIndex])
	p0V,p1V,pSpam=trainNB0(np.array(trainMat),np.array(trainClass))
	errorCount=0
	for docIndex in testSet:
		wordVec=set2Vec(vocablist,doclist[docIndex])
		if classifyNB(np.array(wordVec),p0V,p1V,pSpam)!=classlist[docIndex]:
			errorCount+=1
			print("分类错误的测试集：",doclist[docIndex])
	print('错误率：%.2f%%' %(float(errorCount)/len(testSet)*100))

spamTest()