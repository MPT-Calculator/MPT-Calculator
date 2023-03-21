# Importing
import os
import sys
import time
import math
import multiprocessing as multiprocessing
import tqdm

import cmath
import numpy as np

import netgen.meshing as ngmeshing
from ngsolve import *

sys.path.insert(0, "Functions")
from ..Core_MPT.Theta0 import *
from ..Core_MPT.Theta1 import *
from ..Core_MPT.Theta1_Sweep import *
from ..Core_MPT.MPTCalculator import *

sys.path.insert(0, "Settings")
from Settings import SolverParameters


# Function definition for a full order frequency sweep
def FullSweep(Object, Order, alpha, inorout, mur, sig, Array, BigProblem, NumSolverThreads, Integration_Order, Additional_Int_Order, curve=5, ):
    Object = Object[:-4] + ".vol"
    # Set up the Solver Parameters
    Solver, epsi, Maxsteps, Tolerance, _, _ = SolverParameters()

    # Loading the object file
    ngmesh = ngmeshing.Mesh(dim=3)
    ngmesh.Load("VolFiles/" + Object)

    # Creating the mesh and defining the element types
    mesh = Mesh("VolFiles/" + Object)
    mesh.Curve(curve)  # This can be used to refine the mesh
    numelements = mesh.ne  # Count the number elements
    print(" mesh contains " + str(numelements) + " elements")

    # Set up the coefficients
    # Scalars
    Mu0 = 4 * np.pi * 10 ** (-7)
    NumberofFrequencies = len(Array)
    # Coefficient functions
    mu_coef = [mur[mat] for mat in mesh.GetMaterials()]
    mu = CoefficientFunction(mu_coef)
    inout_coef = [inorout[mat] for mat in mesh.GetMaterials()]
    inout = CoefficientFunction(inout_coef)
    sigma_coef = [sig[mat] for mat in mesh.GetMaterials()]
    sigma = CoefficientFunction(sigma_coef)

    # Set up how the tensor and eigenvalues will be stored
    N0 = np.zeros([3, 3])
    R = np.zeros([3, 3])
    I = np.zeros([3, 3])

    #########################################################################
    # Theta0
    # This section solves the Theta0 problem to calculate both the inputs for
    # the Theta1 problem and calculate the N0 tensor

    # Setup the finite element space
    fes = HCurl(mesh, order=Order, dirichlet="outer", flags={"nograds": True})
    # Count the number of degrees of freedom
    ndof = fes.ndof

    # Define the vectors for the right hand side
    evec = [CoefficientFunction((1, 0, 0)), CoefficientFunction((0, 1, 0)), CoefficientFunction((0, 0, 1))]

    # Setup the grid functions and array which will be used to save
    Theta0i = GridFunction(fes)
    Theta0j = GridFunction(fes)
    Theta0Sol = np.zeros([ndof, 3])

    # Run in three directions and save in an array for later
    for i in tqdm.tqdm(range(3), desc='Solving Theta0'):
        Theta0Sol[:, i] = Theta0(fes, Order, alpha, mu, inout, evec[i], Tolerance, Maxsteps, epsi, i + 1, Solver, Additional_Int_Order)
    print(' solved theta0 problems    ')

    # Calculate the N0 tensor
    VolConstant = Integrate(1 - mu ** (-1), mesh, order=Integration_Order)
    for i in range(3):
        Theta0i.vec.FV().NumPy()[:] = Theta0Sol[:, i]
        for j in range(3):
            Theta0j.vec.FV().NumPy()[:] = Theta0Sol[:, j]
            if i == j:
                N0[i, j] = (alpha ** 3) * (VolConstant + (1 / 4) * (
                    Integrate(mu ** (-1) * (InnerProduct(curl(Theta0i), curl(Theta0j))), mesh, order=Integration_Order)))
            else:
                N0[i, j] = (alpha ** 3 / 4) * (
                    Integrate(mu ** (-1) * (InnerProduct(curl(Theta0i), curl(Theta0j))), mesh, order=Integration_Order))

        # Copy the tensor
        # N0+=np.transpose(N0-np.eye(3)@N0)

    #########################################################################
    # Theta1
    # This section solves the Theta1 problem to calculate the solution vectors
    # of the snapshots

    # Setup the finite element space
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes2 = HCurl(mesh, order=Order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    # Count the number of degrees of freedom
    ndof2 = fes2.ndof

    # Define the vectors for the right hand side
    xivec = [CoefficientFunction((0, -z, y)), CoefficientFunction((z, 0, -x)), CoefficientFunction((-y, x, 0))]

    # Solve the problem
    TensorArray, EigenValues = Theta1_Sweep(Array, mesh, fes, fes2, Theta0Sol, xivec, alpha, sigma, mu, inout,
                                            Tolerance, Maxsteps, epsi, Solver, N0, NumberofFrequencies, False, True,
                                            False, False, Order, NumSolverThreads, Integration_Order, Additional_Int_Order)

    print(' solved theta1 problems     ')
    print(' frequency sweep complete')

    return TensorArray, EigenValues, N0, numelements, (ndof, ndof2)

