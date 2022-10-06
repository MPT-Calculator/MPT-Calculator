#Importing
import os
import sys
import numpy as np

sys.path.insert(0,"Functions")
from Plotters import *


#Retrieve the data

#Frequency arrays
Frequencies = np.genfromtxt("Data/Frequencies.csv",delimiter=",")
PODFrequencies = np.genfromtxt("Data/PODFrequencies.csv",delimiter=",")

#Eigenvalue arrays
Eigenvalues = np.genfromtxt("Data/Eigenvalues.csv",delimiter=",",dtype=complex)
PODEigenvalues = np.genfromtxt("Data/PODEigenvalues.csv",delimiter=",",dtype=complex)

#Tensor coefficient arrays
Tensors = np.genfromtxt("Data/Tensors.csv",delimiter=",",dtype=complex)
PODTensors = np.genfromtxt("Data/PODTensors.csv",delimiter=",",dtype=complex)

#Error Bars
Errors = np.genfromtxt("Data/ErrorBars.csv",delimiter=",")

#Eddy-current breakdown line
try:
    f = open("Data/Eddy-current_breakdown.txt","r")
    exec(f.readline())
    f.close()
    omega = float(omega)
except:
    omega = False



#remove the rows so that the array represents an upper triangular matrix
Tensors = np.concatenate([np.concatenate([Tensors[:,:3],Tensors[:,4:6]],axis=1),Tensors[:,8:9]],axis=1)
PODTensors = np.concatenate([np.concatenate([PODTensors[:,:3],PODTensors[:,4:6]],axis=1),PODTensors[:,8:9]],axis=1)
Errors = np.concatenate([np.concatenate([Errors[:,:3],Errors[:,4:6]],axis=1),Errors[:,8:9]],axis=1)

#define the place to store it This is relative to the current file
savename = "Graphs/"

#plot the graphs
Show = PODEigPlotter(savename,Frequencies,PODFrequencies,Eigenvalues,PODEigenvalues,omega)
Show = PODErrorPlotter(savename,Frequencies,PODFrequencies,Tensors,PODTensors,Errors,omega)

#plot the graph if required
if Show==True:
    plt.show()
    
