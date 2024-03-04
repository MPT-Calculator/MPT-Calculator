"""
This file contains the functions called from the main.py file for a full order frequency sweep
Functions -FullSweep (frequency sweep no pod)
          -FullSweepMulti (frequency sweep in parallel no pod)

EDIT: 06 Aug 2022: James Elgy
Changed how N0 was calculated for FullSweep to be consistent with FullSweepMulti.
"""

#Importing
import os
import sys
import time
import math
import multiprocessing as multiprocessing

import cmath
import numpy as np

import netgen.meshing as ngmeshing
from ngsolve import *

sys.path.insert(0,"Functions")
from MPTFunctions import *
sys.path.insert(0,"Settings")
from Settings import SolverParameters



#Function definition for a full order frequency sweep
def FullSweep(Object,Order,alpha,inorout,mur,sig,Array,BigProblem, curve=5):
    Object = Object[:-4]+".vol"
    #Set up the Solver Parameters
    Solver,epsi,Maxsteps,Tolerance, _, _ = SolverParameters()
    
    #Loading the object file
    ngmesh = ngmeshing.Mesh(dim=3)
    ngmesh.Load("VolFiles/"+Object)
    
    #Creating the mesh and defining the element types
    mesh = Mesh("VolFiles/"+Object)
    mesh.Curve(curve)#This can be used to refine the mesh
    numelements = mesh.ne#Count the number elements
    print(" mesh contains "+str(numelements)+" elements")

    #Set up the coefficients
    #Scalars
    Mu0 = 4*np.pi*10**(-7)
    NumberofFrequencies = len(Array)
    #Coefficient functions
    mu_coef = [ mur[mat] for mat in mesh.GetMaterials() ]
    mu = CoefficientFunction(mu_coef)
    inout_coef = [inorout[mat] for mat in mesh.GetMaterials() ]
    inout = CoefficientFunction(inout_coef)
    sigma_coef = [sig[mat] for mat in mesh.GetMaterials() ]
    sigma = CoefficientFunction(sigma_coef)
    
    #Set up how the tensor and eigenvalues will be stored
    N0=np.zeros([3,3])
    R=np.zeros([3,3])
    I=np.zeros([3,3])



#########################################################################
#Theta0
#This section solves the Theta0 problem to calculate both the inputs for
#the Theta1 problem and calculate the N0 tensor

    #Setup the finite element space
    fes = HCurl(mesh, order=Order, dirichlet="outer", flags = { "nograds" : True })
    #Count the number of degrees of freedom
    ndof = fes.ndof
    
    #Define the vectors for the right hand side
    evec = [ CoefficientFunction( (1,0,0) ), CoefficientFunction( (0,1,0) ), CoefficientFunction( (0,0,1) ) ]
    
    #Setup the grid functions and array which will be used to save
    Theta0i = GridFunction(fes)
    Theta0j = GridFunction(fes)
    Theta0Sol = np.zeros([ndof,3])


    #Run in three directions and save in an array for later
    for i in range(3):
        Theta0Sol[:,i] = Theta0(fes,Order,alpha,mu,inout,evec[i],Tolerance,Maxsteps,epsi,i+1,Solver)
    print(' solved theta0 problems    ')

    #Calculate the N0 tensor
    VolConstant = Integrate(1-mu**(-1),mesh)
    for i in range(3):
        Theta0i.vec.FV().NumPy()[:] = Theta0Sol[:,i]
        for j in range(3):
            Theta0j.vec.FV().NumPy()[:] = Theta0Sol[:,j]
            if i==j:
                N0[i,j] = (alpha**3)*(VolConstant+(1/4)*(Integrate(mu**(-1)*(InnerProduct(curl(Theta0i),curl(Theta0j))),mesh, order=2*(Order+1))))
            else:
                N0[i,j] = (alpha**3/4)*(Integrate(mu**(-1)*(InnerProduct(curl(Theta0i),curl(Theta0j))),mesh, order=2*(Order+1)))

        #Copy the tensor
        # N0+=np.transpose(N0-np.eye(3)@N0)


#########################################################################
#Theta1
#This section solves the Theta1 problem to calculate the solution vectors
#of the snapshots

    #Setup the finite element space
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes2 = HCurl(mesh, order=Order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    #Count the number of degrees of freedom
    ndof2 = fes2.ndof
    
    #Define the vectors for the right hand side
    xivec = [ CoefficientFunction( (0,-z,y) ), CoefficientFunction( (z,0,-x) ), CoefficientFunction( (-y,x,0) ) ]
    
    #Solve the problem
    TensorArray, EigenValues = Theta1_Sweep(Array,mesh,fes,fes2,Theta0Sol,xivec,alpha,sigma,mu,inout,Tolerance,Maxsteps,epsi,Solver,N0,NumberofFrequencies,False,True,False,False)
    
    print(' solved theta1 problems     ')
    print(' frequency sweep complete')
    
    return TensorArray, EigenValues, N0, numelements, (ndof, ndof2)



#Function definition for a full order frequency sweep in parallel
def FullSweepMulti(Object,Order,alpha,inorout,mur,sig,Array,CPUs,BigProblem, curve=5):
    Object = Object[:-4]+".vol"
    #Set up the Solver Parameters
    Solver,epsi,Maxsteps,Tolerance, AdditionalInt, _ = SolverParameters()
    
    #Loading the object file
    ngmesh = ngmeshing.Mesh(dim=3)
    ngmesh.Load("VolFiles/"+Object)
    
    #Creating the mesh and defining the element types
    mesh = Mesh("VolFiles/"+Object)
    mesh.Curve(curve)#This can be used to refine the mesh
    numelements = mesh.ne#Count the number elements
    print(" mesh contains "+str(numelements)+" elements")

    #Set up the coefficients
    #Scalars
    Mu0 = 4*np.pi*10**(-7)
    NumberofFrequencies = len(Array)
    #Coefficient functions
    mu_coef = [ mur[mat] for mat in mesh.GetMaterials() ]
    mu = CoefficientFunction(mu_coef)
    inout_coef = [inorout[mat] for mat in mesh.GetMaterials() ]
    inout = CoefficientFunction(inout_coef)
    sigma_coef = [sig[mat] for mat in mesh.GetMaterials() ]
    sigma = CoefficientFunction(sigma_coef)
    
    #Set up how the tensor and eigenvalues will be stored
    N0=np.zeros([3,3])
    TensorArray=np.zeros([NumberofFrequencies,9], dtype=complex)
    RealEigenvalues = np.zeros([NumberofFrequencies,3])
    ImaginaryEigenvalues = np.zeros([NumberofFrequencies,3])
    EigenValues = np.zeros([NumberofFrequencies,3], dtype=complex)



#########################################################################
#Theta0
#This section solves the Theta0 problem to calculate both the inputs for
#the Theta1 problem and calculate the N0 tensor

    #Setup the finite element space
    fes = HCurl(mesh, order=Order, dirichlet="outer", flags = { "nograds" : True })
    #Count the number of degrees of freedom
    ndof = fes.ndof
    
    #Define the vectors for the right hand side
    evec = [ CoefficientFunction( (1,0,0) ), CoefficientFunction( (0,1,0) ), CoefficientFunction( (0,0,1) ) ]
    
    #Setup the grid functions and array which will be used to save
    Theta0i = GridFunction(fes)
    Theta0j = GridFunction(fes)
    Theta0Sol = np.zeros([ndof,3])
    
    #Setup the inputs for the functions to run
    Theta0CPUs = min(3,multiprocessing.cpu_count(),CPUs)
    Runlist = []
    for i in range(3):
        if Theta0CPUs<3:
            NewInput = (fes,Order,alpha,mu,inout,evec[i],Tolerance,Maxsteps,epsi,i+1,Solver)
        else:
            NewInput = (fes,Order,alpha,mu,inout,evec[i],Tolerance,Maxsteps,epsi,"No Print",Solver)
        Runlist.append(NewInput)
    #Run on the multiple cores
    with multiprocessing.get_context("spawn").Pool(Theta0CPUs) as pool:
        Output = pool.starmap(Theta0, Runlist)
    print(' solved theta0 problems    ')
    
    #Unpack the outputs
    for i,Direction in enumerate(Output):
        Theta0Sol[:,i] = Direction


#Calculate the N0 tensor
    VolConstant = Integrate(1-mu**(-1),mesh)
    for i in range(3):
        Theta0i.vec.FV().NumPy()[:] = Theta0Sol[:,i]
        for j in range(3):
            Theta0j.vec.FV().NumPy()[:] = Theta0Sol[:,j]
            if i==j:
                N0[i,j] = (alpha**3)*(VolConstant+(1/4)*(Integrate(mu**(-1)*(InnerProduct(curl(Theta0i),curl(Theta0j))),mesh, order=2*(Order+1))))
            else:
                N0[i,j] = (alpha**3/4)*(Integrate(mu**(-1)*(InnerProduct(curl(Theta0i),curl(Theta0j))),mesh, order=2*(Order+1)))



#########################################################################
#Theta1
#This section solves the Theta1 problem and saves the solution vectors

    print(' solving theta1')

    #Setup the finite element space
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes2 = HCurl(mesh, order=Order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    #Count the number of degrees of freedom
    ndof2 = fes2.ndof
    
    #Define the vectors for the right hand side
    xivec = [ CoefficientFunction( (0,-z,y) ), CoefficientFunction( (z,0,-x) ), CoefficientFunction( (-y,x,0) ) ]
    
    #Work out where to send each frequency
    Theta1_CPUs = min(NumberofFrequencies,multiprocessing.cpu_count(),CPUs)
    Core_Distribution = []
    Count_Distribution = []
    for i in range(Theta1_CPUs):
        Core_Distribution.append([])
        Count_Distribution.append([])
    
    #Distribute between the cores
    CoreNumber = 0
    count = 1
    for i,Omega in enumerate(Array):
        Core_Distribution[CoreNumber].append(Omega)
        Count_Distribution[CoreNumber].append(i)
        if CoreNumber == CPUs-1 and count == 1:
            count = -1
        elif CoreNumber == 0 and count == -1:
            count = 1
        else:
            CoreNumber +=count
    
    #Create the inputs
    Runlist = []
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)
    for i in range(Theta1_CPUs):
        Runlist.append((Core_Distribution[i],mesh,fes,fes2,Theta0Sol,xivec,alpha,sigma,mu,inout,Tolerance,Maxsteps,epsi,Solver,N0,NumberofFrequencies,False,True,counter,False, Order))
    
    #Run on the multiple cores
    with multiprocessing.get_context("spawn").Pool(Theta1_CPUs) as pool:
        Outputs = pool.starmap(Theta1_Sweep, Runlist)
    
    #Unpack the results
    for i,Output in enumerate(Outputs):
        for j,Num in enumerate(Count_Distribution[i]):
            TensorArray[Num,:] = Output[0][j]
            EigenValues[Num,:] = Output[1][j]
    
    print("Frequency Sweep complete")
    
    return TensorArray, EigenValues, N0, numelements, (ndof, ndof2)


# def set_niceness():
#     # is called at every process start
#     p = psutil.Process(os.getpid())
#     if sys.platform == 'win32':
#         # set to lowest priority, this is windows only, on Unix use ps.nice(19)
#         p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
#     else:
#         p.nice(19)
