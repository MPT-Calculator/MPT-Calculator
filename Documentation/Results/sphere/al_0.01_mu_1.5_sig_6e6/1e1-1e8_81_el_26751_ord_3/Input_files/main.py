########################################################################
#This code was written by Ben Wilson with the supervison of Paul Ledger
#at the ZCCE, Swansea University
#Powered by NETGEN/NGSolve
########################################################################


#User Inputs

#Geometry
Geometry = "sphere.geo"
#(string) Name of the .geo file to be used in the frequency sweep i.e.
# "sphere.geo"


#Scaling to be used in the sweep in meters
alpha = 0.01
#(float) scaling to be applied to the .geo file i.e. if you have defined
#a sphere of unit radius in a .geo file   alpha = 0.01   would simulate a
#sphere with a radius of 0.01m ( or 1cm)


#About the mesh
#How fine should the mesh be
MeshSize = 2
#(int 1-5) this defines how fine the mesh should be for regions that do
#not have maxh values defined for them in the .geo file (1=verycoarse,
#5=veryfine)


#The order of the elements in the mesh
Order = 3
#(int) this defines the order of each of the elements in the mesh 


#About the Frequency sweep (frequencies are in radians per second)
#Minimum frequency (Powers of 10 i.e Start = 2 => 10**2)
Start = 1
#(float)
#Maximum frequency (Powers of 10 i.e Start = 8 => 10**8)
Finish = 8
#(float)
#Number of points in the freqeuncy sweep
Points = 81
#(int) the number of logarithmically spaced points in the sweep


#I only require a single frequency
Single = False
#(boolean) True if single frequency is required
Omega = 133.5
#(float) the frequency to be solved if Single = True

#POD
#I want to use POD in the frequency sweep
Pod = False
#(boolean) True if POD is to be used, the number of snapshots can be
#edited in in the Settings.py file

#Plot the POD points
PlotPod = False
#(boolean) do you want to plot the snapshots (This requires additional
#calculations and will slow down sweep by around 2% for default settings)

#MultiProcessing
MultiProcessing = True
#(boolean) #I have multiple cores at my disposal and have enough spare RAM
# to run the frequency sweep in parallel (Edit the number of cores to be
#used in the Settings.py file)








########################################################################


#Main script


#Importing
import sys
import numpy as np
sys.path.insert(0,"Functions")
sys.path.insert(0,"Settings")
from MeshCreation import *
from Settings import DefaultSettings
from SingleSolve import SingleFrequency
from FullSolvers import *
from PODSolvers import *
from ResultsFunctions import *
import time

start = time.time()


#Meshing
#Create the mesh
Meshmaker(Geometry,MeshSize)
#Update the .vol file and create the material dictionaries
Materials,mur,sig,inorout = VolMatUpdater(Geometry)

#create the array of points to be used in the sweep
Array = np.logspace(Start,Finish,Points)
CPUs,PODPoints,PODTol,PODErrorBars = DefaultSettings()
SavePOD = False
if PODErrorBars!=True:
    ErrorTensors=False
else:
    ErrorTensors=True
PODArray = np.logspace(Start,Finish,PODPoints)
#PODArray = np.linspace(10**Start,10**Finish,PODPoints)

#Create the folders which will be used to save everything
sweepname = FolderMaker(Geometry, Single, Array, Omega, Pod, PlotPod, PODArray, PODTol, alpha, Order, MeshSize, mur, sig, ErrorTensors)

#Run the sweep

if Single==True:
    if MultiProcessing!=True:
        CPUs = 1
    MPT, EigenValues, N0, elements = SingleFrequency(Geometry,Order,alpha,inorout,mur,sig,Omega,CPUs)
else:
    if Pod==True:
        if MultiProcessing==True:
            if PlotPod==True:
                if PODErrorBars==True:
                    TensorArray, EigenValues, N0, PODTensors, PODEigenValues, elements, ErrorTensors = PODSweepMulti(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,CPUs,sweepname,SavePOD,PODErrorBars)
                else:
                    TensorArray, EigenValues, N0, PODTensors, PODEigenValues, elements = PODSweepMulti(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,CPUs,sweepname,SavePOD,PODErrorBars)
            else:
                if PODErrorBars==True:
                    TensorArray, EigenValues, N0, elements, ErrorTensors = PODSweepMulti(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,CPUs,sweepname,SavePOD,PODErrorBars)
                else:
                    TensorArray, EigenValues, N0, elements = PODSweepMulti(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,CPUs,sweepname,SavePOD,PODErrorBars)
        else:
            if PlotPod==True:
                if PODErrorBars==True:
                    TensorArray, EigenValues, N0, PODTensors, PODEigenValues, elements, ErrorTensors = PODSweep(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,sweepname,SavePOD,PODErrorBars)
                else:
                    TensorArray, EigenValues, N0, PODTensors, PODEigenValues, elements = PODSweep(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,sweepname,SavePOD,PODErrorBars)
            else:
                if PODErrorBars==True:
                    TensorArray, EigenValues, N0, elements, ErrorTensors = PODSweep(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,sweepname,SavePOD,PODErrorBars)
                else:
                    TensorArray, EigenValues, N0, elements = PODSweep(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,sweepname,SavePOD,PODErrorBars)
    else:
        if MultiProcessing==True:
            TensorArray, EigenValues, N0, elements = FullSweepMulti(Geometry,Order,alpha,inorout,mur,sig,Array,CPUs)
        else:
            TensorArray, EigenValues, N0, elements = FullSweep(Geometry,Order,alpha,inorout,mur,sig,Array)


#Plotting and saving

if Single==True:
    SingleSave(Geometry, Omega, MPT, EigenValues, N0, elements, alpha, Order, MeshSize, mur, sig)
elif PlotPod==True:
    if Pod==True:
        PODSave(Geometry, Array, TensorArray, EigenValues, N0, PODTensors, PODEigenValues, PODArray, PODTol, elements, alpha, Order, MeshSize, mur, sig, ErrorTensors)
    else:
        FullSave(Geometry, Array, TensorArray, EigenValues, N0, Pod, PODArray, PODTol, elements, alpha, Order, MeshSize, mur, sig, ErrorTensors)
else:
    FullSave(Geometry, Array, TensorArray, EigenValues, N0, Pod, PODArray, PODTol, elements, alpha, Order, MeshSize, mur, sig, ErrorTensors)


print(time.time()-start)

