import numpy as np
from numpy import linalg as linalg

class KalmanFilter:

	def __init__(self):
		#Parameters
		self.F=None#State-transition model
		self.B=None#Control-model
		self.Q=None#Process-noise covariance
		self.H=None#Observation model
		self.R=None#Output noise covariance
		#Outputs
		self.P=None#Prediction covariance matrix
		self.x=None#Prediction
		self.innovation=None
		self.residual=None
		self.y=None#Observed vector
		self.S=None
	def __init__(self, F, B, Q, H, R, P, x):

		self.F=F#State-transition model
		self.B=B#Control-model
		self.Q=Q#Process-noise covariance
		self.H=H#Observation model
		self.R=R#Output noise covariance
		#Outputs
		self.P=P#Prediction covariance matrix
		self.x=x#Prediction
		self.innovation=np.zeros(np.shape(x))
		self.residual=np.zeros(np.shape(x))
		self.y=np.zeros(np.shape(x))#Observed vector
		self.S=np.zeros(np.shape(R))

	def predict(self, u):
		
		self.x=self.F.dot(self.x)+self.B.dot(u)
		self.P=(self.F.dot(self.P)).dot(self.F.transpose())+self.Q

	def getPrediction(self):
		
		return self.x, self.P

	def update(self, z):
		
		self.innovation=z-self.H.dot(self.x)
		self.S=self.R+(self.H.dot(self.P)).dot(self.H.transpose())
		self.K=(self.P.dot(self.H.transpose())).dot(linalg.matrix_power(self.S, -1))
		self.x=self.x+self.K.dot(self.innovation)
		self.P=(np.eye(np.shape(self.K)[0])-self.K.dot(self.H)).dot(self.P)
		self.residual=z-self.H.dot(self.x)
