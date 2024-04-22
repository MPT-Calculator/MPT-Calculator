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
from ..Core_MPT.MPT_Preallocation import *
from ..Core_MPT.Solve_Theta_0_Problem import *
from ..Core_MPT.Calculate_N0 import *
from ..Core_MPT.Theta0_Postprocessing import *
from ..Core_MPT.Mat_Method_Calc_Imag_Part import *
from ..Core_MPT.Mat_Method_Calc_Real_Part import *

sys.path.insert(0, "Settings")
from Settings import SolverParameters


# Function definition for a full order frequency sweep
def FullSweep(Object, Order, alpha, inorout, mur, sig, Array, BigProblem, NumSolverThreads, Integration_Order,
              Additional_Int_Order, Order_L2, sweepname, drop_tol, curve=5):
    """
    B.A. Wilson, J.Elgy, P.D.Ledger 2020-2024
    Function to compute MPT for an array of frequencies.
    
    1) Preallocate mesh, finite element spaces, material properties and assign bonus integration orders.
    2) Compute theta0 and N0
    3) Compute theta1 for each frequency in Array.
    4) Compute tensor coefficients. 

    Args:
        Object (str): Geometry file name
        Order (int): order of finite element space.
        alpha (float): object size scaling
        inorout (dict): dictionary of material names that is 1 inside object and 0 outside
        mur (dict): dictionary of mur in each region
        sig (dict): dictionary of sigma in each region
        Array (list | np.ndarray): list of N frequencies (rad/s) to condider.
        BigProblem (bool): flag that problem is large. Will run in a slower but more memory efficient mode.
        NumSolverThreads (str | int): Number of parallel threads to use in iterative solver. If 'default' use all threads.
        Integration_Order (int): order of integration to be used when computing tensors.
        Additional_Int_Order (int): additional orders to be considered when assembling linear and bilinear forms. For use with curved elements adn prisms.
        Order_L2 (int): Order of L2 projection of material coefficient functions onto the mesh to acount for material discontinuities that don't align with mesh.
        sweepname (str): Name of the simulation to be run.
        drop_tol (float | None): Tolerance below which entries in the sparse matrices are assumed to be 0.
        curve (int, optional): Order of polynomial used to approximate curved surfaces. Defaults to 5.

    Returns:
        TensorArray (np.ndarray): Nx9 complex tensor coefficients
        EigenValues (np.ndarray): Nx3 complex eigenvalues
        N0 (np.ndarray): 3x3 N0 tensor,
        numelements (int): nnumber of elements in mesh
        (ndof, ndof2) (tuple): ndof in fes1 and fes2.
    """
    
    

    print(' Running as full sweep')

    EigenValues, Mu0, N0, NumberofFrequencies, _, TensorArray, inout, mesh, mu_inv, numelements, sigma, bilinear_bonus_intorder = MPT_Preallocation(
        Array, Object, [], curve, inorout, mur, sig, Order, Order_L2, sweepname,NumSolverThreads, drop_tol )
    # Set up the Solver Parameters
    Solver, epsi, Maxsteps, Tolerance, _, use_integral = SolverParameters()

    # Set up how the tensor and eigenvalues will be stored
    N0 = np.zeros([3, 3])
    R = np.zeros([3, 3])
    I = np.zeros([3, 3])

    #########################################################################
    # Theta0
    # This section solves the Theta0 problem to calculate both the inputs for
    # the Theta1 problem and calculate the N0 tensor
    # Here, we have set the solver not to use the recovery mode.
    Theta0Sol, Theta0i, Theta0j, fes, ndof, evec = Solve_Theta_0_Problem(Additional_Int_Order, 1, Maxsteps, Order,
                                                                         Solver,
                                                                         Tolerance, alpha, epsi, inout, mesh, mu_inv,
                                                                         False, '')

    print(' solved theta0 problems    ')

    # Poission Projection to account for gradient terms:
    Theta0Sol = Theta0_Postprocessing(Additional_Int_Order, Theta0Sol, fes)

    # Calculate the N0 tensor
    N0 = Calculate_N0(Integration_Order, N0, Theta0Sol, Theta0i, Theta0j, alpha, mesh, mu_inv)


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

     #Solve the problem
    if use_integral is True:
        TensorArray, EigenValues = Theta1_Sweep(Array, mesh, fes, fes2, Theta0Sol, xivec, alpha, sigma, mu_inv, inout,
                                                Tolerance, Maxsteps, epsi, Solver, N0, NumberofFrequencies, False, True,
                                                False, False, Order, NumSolverThreads, Integration_Order, Additional_Int_Order, bilinear_bonus_intorder, drop_tol)
    else:
        Theta1Sols = Theta1_Sweep(Array, mesh, fes, fes2, Theta0Sol, xivec, alpha, sigma, mu_inv, inout,
                                                Tolerance, Maxsteps, epsi, Solver, N0, NumberofFrequencies, True, False,
                                                False, False, Order, NumSolverThreads, Integration_Order, Additional_Int_Order, bilinear_bonus_intorder, drop_tol)


        
        U_proxy = sp.eye(fes2.ndof)

        real_part = Mat_Method_Calc_Real_Part(bilinear_bonus_intorder, fes2, inout, mu_inv, alpha, np.squeeze(np.asarray(Theta1Sols)),
            U_proxy, U_proxy, U_proxy, NumSolverThreads, drop_tol, BigProblem, ReducedSolve=False)

        imag_part = Mat_Method_Calc_Imag_Part(Array, Integration_Order, Theta0Sol, bilinear_bonus_intorder, fes2, mesh, inout, alpha, np.squeeze(np.asarray(Theta1Sols)),
            sigma, U_proxy, U_proxy, U_proxy, xivec,  NumSolverThreads, drop_tol, BigProblem, ReducedSolve=False)
        
        for Num in range(len(Array)):
            TensorArray[Num, :] = real_part[Num,:] + N0.flatten()
            TensorArray[Num, :] += 1j * imag_part[Num,:]

            R = TensorArray[Num, :].real.reshape(3, 3)
            I = TensorArray[Num, :].imag.reshape(3, 3)
            EigenValues[Num, :] = np.sort(np.linalg.eigvals(R)) + 1j * np.sort(np.linalg.eigvals(I))
    
    
    print(' solved theta1 problems     ')

    # if use_integral is False:
    #     # I'm aware that pre and post multiplying by identity of size ndof2 is slower than using K and A matrices outright,
    #     # however this allows us to reuse the Construct_Matrices function rather than add (significantly) more code.
    #     identity1 = sp.identity(ndof2)
    #     TensorArray, EigenValues = Theta1_Lower_Sweep(Array, mesh, fes, fes2, Theta1Sols, identity1, identity1, identity1,
    #                                Theta0Sol, xivec, alpha, sigma, mu_inv, inout, N0, NumberofFrequencies, False,
    #                                False, 0, 0, Order, Integration_Order, bilinear_bonus_intorder, use_integral)

    print(' frequency sweep complete')

    return TensorArray, EigenValues, N0, numelements, (ndof, ndof2)

