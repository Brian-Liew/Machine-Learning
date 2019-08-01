import numpy as np
import matplotlib.pyplot as plt 
x = np.arange(0,5,0.01)
y = -3*x**2+8*x
plt.figure(1)
plt.plot(x,y)

def Gradient_As():
	#求导函数，根据不同函数可以写出不同的求导函数
	def derivation(x_old):
		return -6*x_old+8
	#设定初始值、步长、精度
	x_old=-2
	x_new=0
	alpha=0.01
	presision=0.00000001
	#爬坡
	while abs(x_new-x_old)>presision:
		x_old=x_new
		x_new=x_old+alpha*derivation(x_old)
		plt.figure(1)
		plt.scatter(x_new,-3*x_new**2+8*x_new,color='r',marker='+')
	plt.show()
	print(x_new)
	
Gradient_As()