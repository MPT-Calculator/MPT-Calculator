#Importing
import os
import sys
import numpy as np

sys.path.insert(0,"Functions")
from Saving.EigPlotter import *
from Saving.TensorPlotter import *


#Retrieve the data

#Frequency arrays
Frequencies = np.genfromtxt("Data/Frequencies.csv",delimiter=",")

#Eigenvalue arrays
Eigenvalues = np.genfromtxt("Data/Eigenvalues.csv",delimiter=",",dtype=complex)

#Tensor coefficient arrays
Tensors = np.genfromtxt("Data/Tensors.csv",delimiter=",",dtype=complex)

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

#define the place to store it This is relative to the current file
savename = "Graphs/"

#plot the graphs
Show = EigPlotter(savename,Frequencies,Eigenvalues,omega)
Show = TensorPlotter(savename,Frequencies,Tensors,omega)

#plot the graph if required
if Show==True:
    plt.show()



