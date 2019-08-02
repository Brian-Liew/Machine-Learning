from sklearn.linear_model import LogisticRegression

def colicTest():
	frTrain = open('horseColicTraining.txt')
	frTest = open('horseColicTest.txt')
	trainingSet =[]
	traininngLabels=[]
	testSet=[]
	testLabels=[]
	for line in frTrain.readlines() :
		currLine = line.strip().split('\t')
		lineArr=[]
		for  i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		traininngLabels.append(float(currLine[-1]))
	classifier=LogisticRegression(solver='liblinear',max_iter=10).fit(trainingSet,traininngLabels)
	for line in frTest.readlines():
		currLine =line.strip().split('\t')
		lineArr=[]
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		testSet.append(lineArr)
		testLabels.append(float(currLine[-1]))
	test_accurcy=classifier.score(testSet,testLabels)*100
	print("测试准确率率为：%.2f%%" % test_accurcy)

colicTest()