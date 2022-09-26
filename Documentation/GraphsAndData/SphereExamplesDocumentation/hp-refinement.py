#Importing
import math
import cmath
import numpy as np
import csv



Omega = 133.5
Alpha = 0.01
Mur = 1.5
Sigma = 6*10**6


Mu0=4*np.pi*10**(-7)
nu=Omega*Mu0*Sigma*(Alpha**2)
k=cmath.sqrt(1j*Mur*Mu0*Sigma*Omega)
v=k*Alpha;
Im12=cmath.sqrt(2/np.pi/v)*np.cosh(v)
Ip12=cmath.sqrt(2/np.pi/v)*np.sinh(v)
m=2*np.pi*(Alpha**3)*( (2*(Mur*Mu0)+(Mu0))*v*Im12-(Mu0*(1+(v**2))+2*(Mur*Mu0))*Ip12)/(((Mur*Mu0)-Mu0)*v*Im12+((Mu0)*(1+(v**2))-(Mur*Mu0))*Ip12)
m=m-(2*m.imag*1j)


ActualMPT = np.eye(3)*m
print("The exact tensor is",ActualMPT)

#Retrieve the MPT
TestMPT = np.genfromtxt("Data/MPT.csv",delimiter=",",dtype=complex)
print("The sample tensor is",TestMPT)
ErrorMPT = (abs(TestMPT-ActualMPT))**2
print("The error tensor is",ErrorMPT)
Error = np.sum(ErrorMPT)**(1/2)
print("The error is",Error)
RelativeError = Error/(np.sum(abs(ActualMPT)**2)**(1/2))
print("The relative error is",RelativeError)
