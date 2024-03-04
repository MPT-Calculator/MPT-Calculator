#This file contains the function called from the main.py file when Single=True
#Functions -SingleFrequency (Solve for one value of omega)
#Importing
import os
import sys
import time
import multiprocessing as multiprocessing

import cmath
import numpy as np

import netgen.meshing as ngmeshing
from ngsolve import *

sys.path.insert(0,"Functions")
from MPTFunctions import *
from ResultsFunctions import FtoS
sys.path.insert(0,"Settings")
from Settings import SolverParameters
import integration_test
import gc

from Functions.Helper_Functions.count_prismatic_elements import count_prismatic_elements


def SingleFrequency(Object,Order,alpha,inorout,mur,sig,Omega,CPUs,VTK,Refine, curve=5, theta_solutions_only=False):
    Object = Object[:-4]+".vol"
    #Set up the Solver Parameters
    Solver,epsi,Maxsteps,Tolerance,AdditionalInt, _ = SolverParameters()

    N_prisms, N_tets = count_prismatic_elements('./VolFiles/' + Object)
    if N_prisms > 0:
        prism_flag = True
    else:
        prism_flag = False

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
    # Mu0 = 4*np.longdouble('3.141592653589793238462643383279502884')*(10**-7)
    #Coefficient functions
    mu_coef = [ mur[mat] for mat in mesh.GetMaterials() ]
    mu = CoefficientFunction(mu_coef)
    inout_coef = [inorout[mat] for mat in mesh.GetMaterials() ]
    inout = CoefficientFunction(inout_coef)
    sigma_coef = [sig[mat] for mat in mesh.GetMaterials() ]
    sigma = CoefficientFunction(sigma_coef)

    #Set up how the tensors will be stored
    N0 = np.zeros([3,3])
    R = np.zeros([3,3])
    I = np.zeros([3,3])


#########################################################################
#Theta0
#This section solves the Theta0 problem to calculate both the inputs for
#the Theta1 problem and calculate the N0 tensor

    #Setup the finite element space
    # fes = HCurl(mesh, order=Order, dirichlet="outer", flags = { "nograds" : True })

    # For testing, I'm making the t0 space the same size as the t1 space.
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes = HCurl(mesh, order=Order, dirichlet="outer", complex=False, gradientdomains=dom_nrs_metal)

    #Count the number of degrees of freedom
    ndof = fes.ndof

    #Define the vectors for the right hand side
    evec = [ CoefficientFunction( (1,0,0) ), CoefficientFunction( (0,1,0) ), CoefficientFunction( (0,0,1) ) ]

    #Setup the grid functions and array which will be used to save
    Theta0i = GridFunction(fes)
    Theta0i.Set((0,0,0))
    Theta0j = GridFunction(fes)
    Theta0j.Set((0,0,0))
    Theta0Sol = np.zeros([ndof,3], dtype=np.longdouble)

    #Setup the inputs for the functions to run
    Runlist = []
    for i in range(3):
        if CPUs<3:
            NewInput = (fes,Order,alpha,mu,inout,evec[i],Tolerance,Maxsteps,epsi,i+1,Solver)
        else:
            NewInput = (fes,Order,alpha,mu,inout,evec[i],Tolerance,Maxsteps,epsi,"No Count",Solver)
        Runlist.append(NewInput)
    #Run on the multiple cores
    with multiprocessing.Pool(CPUs) as pool:
        Output = pool.starmap(Theta0, Runlist)
    print(' solved theta0 problem      ')

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
                N0[i,j] = (alpha**3)*(VolConstant+(1/4)*(Integrate(mu**(-1)*(InnerProduct(curl(Theta0i),curl(Theta0j))),mesh, order=2*Order)))
            else:
                N0[i,j] = (alpha**3/4)*(Integrate(mu**(-1)*(InnerProduct(curl(Theta0i),curl(Theta0j))),mesh, order=2*Order))

# FOR TESTING: Removing gradient terms introduced by making fes the same size as fes2:
    # Poission Projection to account for gradient terms:
    u, v = fes.TnT()
    m = BilinearForm(fes)
    m += u * v * dx
    m.Assemble()

    # build gradient matrix as sparse matrix (and corresponding scalar FESpace)
    gradmat, fesh1 = fes.CreateGradient()

    gradmattrans = gradmat.CreateTranspose()  # transpose sparse matrix
    math1 = gradmattrans @ m.mat @ gradmat  # multiply matrices
    math1[0, 0] += 1  # fix the 1-dim kernel
    invh1 = math1.Inverse(inverse="sparsecholesky")

    # build the Poisson projector with operator Algebra:
    proj = IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ m.mat
    theta0 = GridFunction(fes)
    for i in range(3):
        theta0.vec.FV().NumPy()[:] = Theta0Sol[:, i]
        theta0.vec.data = proj * (theta0.vec)
        Theta0Sol[:, i] = theta0.vec.FV().NumPy()[:]

#########################################################################
#Theta1
#This section solves the Theta1 problem and saves the solution vectors

    #Setup the finite element space
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes2 = HCurl(mesh, order=Order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    #Count the number of degrees of freedom
    ndof2 = fes2.ndof
    
    #Define the vectors for the right hand side
    xivec = [ CoefficientFunction( (0,-z,y) ), CoefficientFunction( (z,0,-x) ), CoefficientFunction( (-y,x,0) ) ]

    #Setup the array which will be used to store the solution vectors
    Theta1Sol = np.zeros([ndof2,3],dtype=complex)
    
    #Set up the inputs for the problem
    Runlist = []
    nu = Omega*Mu0*(alpha**2)
    for i in range(3):
        if CPUs<3:
            NewInput = (fes,fes2,Theta0Sol[:,i],xivec[i],Order,alpha,nu,sigma,mu,inout,Tolerance,Maxsteps,epsi,Omega,i+1,3,Solver)
        else:
            NewInput = (fes,fes2,Theta0Sol[:,i],xivec[i],Order,alpha,nu,sigma,mu,inout,Tolerance,Maxsteps,epsi,Omega,"No Print",3,Solver)
        Runlist.append(NewInput)
    
    #Run on the multiple cores
    with multiprocessing.Pool(CPUs) as pool:
        Output = pool.starmap(Theta1, Runlist)
    print(' solved theta1 problem       ')
    
    
    #Unpack the outputs
    for i, OutputNumber in enumerate(Output):
        Theta1Sol[:,i] = OutputNumber

    if theta_solutions_only == True:
        return Theta0Sol, Theta1Sol

    #Create the VTK output if required
    if VTK == True:
        print(' creating vtk output', end='\r')
        ThetaE1 = GridFunction(fes2)
        ThetaE2 = GridFunction(fes2)
        ThetaE3 = GridFunction(fes2)
        ThetaE1.vec.FV().NumPy()[:] = Output[0]
        ThetaE2.vec.FV().NumPy()[:] = Output[1]
        ThetaE3.vec.FV().NumPy()[:] = Output[2]
        E1Mag = CoefficientFunction(sqrt(InnerProduct(ThetaE1.real,ThetaE1.real)+InnerProduct(ThetaE1.imag,ThetaE1.imag)))
        E2Mag = CoefficientFunction(sqrt(InnerProduct(ThetaE2.real,ThetaE2.real)+InnerProduct(ThetaE2.imag,ThetaE2.imag)))
        E3Mag = CoefficientFunction(sqrt(InnerProduct(ThetaE3.real,ThetaE3.real)+InnerProduct(ThetaE3.imag,ThetaE3.imag)))
        Sols = []
        Sols.append(dom_nrs_metal)
        Sols.append((ThetaE1*1j*Omega*sigma).real)
        Sols.append((ThetaE1*1j*Omega*sigma).imag)
        Sols.append((ThetaE2*1j*Omega*sigma).real)
        Sols.append((ThetaE2*1j*Omega*sigma).imag)
        Sols.append((ThetaE3*1j*Omega*sigma).real)
        Sols.append((ThetaE3*1j*Omega*sigma).imag)
        Sols.append(E1Mag*Omega*sigma)
        Sols.append(E2Mag*Omega*sigma)
        Sols.append(E3Mag*Omega*sigma)
        savename = "Results/vtk_output/"+Object[:-4]+"/om_"+FtoS(Omega)+"/"
        if Refine == True:
            vtk = VTKOutput(ma=mesh, coefs=Sols, names = ["Object","E1real","E1imag","E2real","E2imag","E3real","E3imag","E1Mag","E2Mag","E3Mag"],filename=savename+Object[:-4],subdivision=3)
        else:
            vtk = VTKOutput(ma=mesh, coefs=Sols, names = ["Object","E1real","E1imag","E2real","E2imag","E3real","E3imag","E1Mag","E2Mag","E3Mag"],filename=savename+Object[:-4],subdivision=0)
        vtk.Do()
        print(' vtk output created     ')
    

    
#########################################################################
#Calculate the tensor and eigenvalues

    #Create the inputs for the calculation of the tensors
    print(' calculating the tensor  ', end='\r')
    Runlist = []
    nu = Omega*Mu0*(alpha**2)
    R,I = MPTCalculator(mesh,fes,fes2,Theta1Sol[:,0],Theta1Sol[:,1],Theta1Sol[:,2],Theta0Sol,xivec,alpha,mu,sigma,inout,nu,"No Print",1, Order)
    print(' calculated the tensor             ') 


    ### Running Integration Test for single freq:
    i = 0
    j = 0
    Theta1i = GridFunction(fes2)
    Theta1i.Set((0,0,0))
    Theta1i.vec.FV().NumPy()[:] = Theta1Sol[:,i]
    Theta1j = GridFunction(fes2)
    Theta1j.Set((0,0,0))
    Theta1j.vec.FV().NumPy()[:] = Theta1Sol[:, j]

    Theta0i = GridFunction(fes2)
    Theta0i.Set((0,0,0))
    Theta0i.vec.FV().NumPy()[:] = Theta0Sol[:,i]
    Theta0j = GridFunction(fes2)
    Theta0j.Set((0,0,0))
    Theta0j.vec.FV().NumPy()[:] = Theta0Sol[:, j]

    np.save('Theta0Sols.npy', Theta0Sol)
    np.save('Theta1Sols.npy', Theta1Sol)

    # integration_test.MassTest(inout, nu, sigma, Theta1i, Theta1j, mesh, fes2)
    # integration_test.MassTest(inout, nu, sigma, Theta1Sol[:,i],Theta1Sol[:,j], mesh, fes2, inorout, sigma_coef, prism_flag=prism_flag)
    # integration_test.ETest(inout, nu, sigma, Theta1Sol[:,i], Theta1Sol[:, j], Theta0Sol[:, i], Theta0Sol[:, j], mesh, fes2, xivec, i, j, prism_flag=prism_flag)

    #Unpack the outputs
    MPT = N0+R+1j*I
    RealEigenvalues = np.sort(np.linalg.eigvals(N0+R))
    ImaginaryEigenvalues = np.sort(np.linalg.eigvals(I))
    EigenValues=RealEigenvalues+1j*ImaginaryEigenvalues

    del Theta1i, Theta1j, Theta0i, Theta0j, fes, fes2, Theta0Sol, Theta1Sol
    gc.collect()

    return MPT, EigenValues, N0, numelements, (ndof, ndof2)
