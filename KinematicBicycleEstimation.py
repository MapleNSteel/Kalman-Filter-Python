import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt
from EKF import ExtendedKalmanFilter

from numpy import tan,arctan,sin,cos

deltaTime=0.01
numSteps=5000

L=3.3138
Lr=L/2
Lf=L/2

def h(x):
	return x
def f(x,u):
	xDot=np.array(np.zeros((1,3)))

	beta=arctan((Lr/(Lf+Lr))*tan(u[1]))

	xDot[0,0]=u[0]*cos(x[2]+beta)
	xDot[0,1]=u[0]*sin(x[2]+beta)	
	xDot[0,2]=(u[0]/Lr)*sin(beta)

	return (x+ xDot*deltaTime)[0]

def jacobianF(x,u):

	beta=arctan((Lr/(Lf+Lr))*tan(u[1]))

	return np.array(np.eye(3))+np.array([[0,0,-u[0]*sin(x[2]+beta)*(u[0]/Lr)*sin(beta)],[0,0,u[0]*cos(x[2]+beta)*(u[0]/Lr)*sin(beta)],[0,0,0]])

def jacobianH(x):
	return np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])

def main():

	F=jacobianF
	H=jacobianH
	P=np.array([[0, 0, 0],[0, 0, 0],[0, 0, 0]])
	
	p_sigma=1e-5
	o_sigma=1e-5	

	Q=np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])*(p_sigma**2)
	R=np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])*(o_sigma**2)

	t=(np.arange(0,numSteps)*0.01)
	i=np.zeros((2,numSteps))
	i[0,0:]=1
	i[1,0:]=np.sin(2*np.pi*0.1*t)*np.pi/6

	p_n=np.reshape(np.random.multivariate_normal([0, 0, 0],[[p_sigma,0,0],[0,p_sigma,0],[0,0,p_sigma]],numSteps),(3,numSteps))
	o_n=np.reshape(np.random.multivariate_normal([0, 0, 0],[[o_sigma,0,0],[0,o_sigma,0],[0,0,o_sigma]],numSteps),(3,numSteps))

	x=np.zeros((3,numSteps))#process output
	y=np.zeros((3,numSteps))#process output
	x_pred=np.zeros((3,numSteps))

	EKF=ExtendedKalmanFilter(jacobianF,Q,jacobianH,R,P,f,h,x[0:,0])

	for t in range(1,numSteps):
		x[0:,t]=f(x[0:,t-1],i[0:,t])+p_n[0:,t]#process output with process noise
		EKF.predict(i[0:,t])
		x_pred[0:,t],P=EKF.getPrediction()#predicting next process state
		#print(P)
		y[0:,t]=h(x[0:,t])+o_n[0:,t]#sensor output
		EKF.update(y[0:,t])#update Kalman
	
	print(np.sum((y-x_pred)**2)/numSteps)
	print(np.sum((x-x_pred)**2)/numSteps)

	plt.plot(x[1,0:],x[0,0:],'r')
	plt.plot(y[1,0:],y[0,0:],'b')
	plt.plot(x_pred[1,0:],x_pred[0,0:],'g')
	plt.show()

if __name__ == '__main__':
	main()
