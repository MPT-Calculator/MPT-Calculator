import os
import sys
import time
import math
import multiprocessing as multiprocessing
from tqdm import tqdm_notebook as tqdm
import cmath
import numpy as np

import netgen.meshing as ngmeshing
from ngsolve import *
import scipy.sparse as sp

sys.path.insert(0, "Functions")
from ..Core_MPT.Theta0 import *
from ..Core_MPT.Theta1 import *
from ..Core_MPT.Theta1_Sweep import *
from ..Core_MPT.MPTCalculator import *
from ..Core_MPT.imap_execution import *
from ..Core_MPT.supress_stdout import *
from ..Core_MPT.MPT_Preallocation import *
from ..Core_MPT.Solve_Theta_0_Problem import *
from ..Core_MPT.Calculate_N0 import *
from ..Core_MPT.Theta0_Postprocessing import *
from ..Core_MPT.Mat_Method_Calc_Imag_Part import *
from ..Core_MPT.Mat_Method_Calc_Real_Part import *

sys.path.insert(0, "Settings")
from Settings import SolverParameters



# Function definition for a full order frequency sweep in parallel
def FullSweepMulti(Object ,Order ,alpha ,inorout ,mur ,sig ,Array ,CPUs ,BigProblem, NumSolverThreads,Integration_Order, Additional_Int_Order, Order_L2,
                   sweepname, drop_tol, curve=5):

    print(' Running as parallel full sweep')


    EigenValues, Mu0, N0, NumberofFrequencies, _, TensorArray, inout, mesh, mu_inv, numelements, sigma, bilinear_bonus_int_order = MPT_Preallocation(
        Array, Object, [], curve, inorout, mur, sig, Order, Order_L2, sweepname,NumSolverThreads, drop_tol )
    # Set up the Solver Parameters
    Solver, epsi, Maxsteps, Tolerance, _, use_integral = SolverParameters()

    #########################################################################
    # Theta0
    # This section solves the Theta0 problem to calculate both the inputs for
    # the Theta1 problem and calculate the N0 tensor

    # Setup the finite element space
    # Here, we either load theta0 or calculate.
    Theta0Sol, Theta0i, Theta0j, fes, ndof, evec = Solve_Theta_0_Problem(Additional_Int_Order, CPUs, Maxsteps, Order,
                                                                         Solver,
                                                                         Tolerance, alpha, epsi, inout, mesh, mu_inv,
                                                                         False, '')

    # Poission Projection to acount for gradient terms:
    Theta0Sol = Theta0_Postprocessing(Additional_Int_Order, Theta0Sol, fes)

    # Calculate the N0 tensor
    N0 = Calculate_N0(Integration_Order, N0, Theta0Sol, Theta0i, Theta0j, alpha, mesh, mu_inv)

    #########################################################################
    # Theta1
    # This section solves the Theta1 problem and saves the solution vectors

    print(' solving theta1')

    # Setup the finite element space
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes2 = HCurl(mesh, order=Order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    # Count the number of degrees of freedom
    ndof2 = fes2.ndof

    # Define the vectors for the right hand side
    xivec = [CoefficientFunction((0, -z, y)), CoefficientFunction((z, 0, -x)), CoefficientFunction((-y, x, 0))]

    # Work out where to send each frequency
    Theta1_CPUs = min(NumberofFrequencies, multiprocessing.cpu_count(), CPUs)

    if use_integral is True:
        vectors = False
        tensors = True
    else:
        vectors = True
        tensors = False
                # Unpack the results
        if BigProblem == True:
            Theta1Sols = np.zeros([ndof2, len(Array), 3], dtype=np.complex64)
        else:
            Theta1Sols = np.zeros([ndof2, len(Array), 3], dtype=complex)

    # Create the inputs
    Runlist = []
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)
    for i in range(len(Array)):
        Runlist.append((np.asarray([Array[i]]), mesh, fes, fes2, Theta0Sol, xivec, alpha, sigma, mu_inv, inout, Tolerance,
                        Maxsteps, epsi, Solver, N0, NumberofFrequencies, vectors, tensors, counter, False, Order, NumSolverThreads, Integration_Order,
                        Additional_Int_Order, bilinear_bonus_int_order, drop_tol, 'Theta1_Sweep'))


    tqdm.tqdm.set_lock(multiprocessing.RLock())
    # Run on the multiple cores
    if ngsglobals.msg_level != 0:
        to = sys.stdout
    else:
        to = os.devnull
    with supress_stdout(to=to):
        with multiprocessing.get_context("spawn").Pool(Theta1_CPUs, maxtasksperchild=1, initializer=tqdm.tqdm.set_lock, initargs=(tqdm.tqdm.get_lock(),)) as pool:
            Outputs = list(tqdm.tqdm(pool.imap(imap_version, Runlist, chunksize=1), total=len(Runlist), desc='Solving Theta1', dynamic_ncols=True))

    # Unpack the results
    
    for i in range(len(Outputs)):
        if vectors is False:
            EigenValues[i,:] = Outputs[i][1][0]
            TensorArray[i,:] = Outputs[i][0][0]
        else:
            for j in range(ndof2):
                Theta1Sols[j,i,:] = Outputs[i][j][0][:]

            
    
    U_proxy = sp.eye(fes2.ndof)
    real_part = Mat_Method_Calc_Real_Part(bilinear_bonus_int_order, fes2, inout, mu_inv, alpha, np.squeeze(np.asarray(Theta1Sols)),
        U_proxy, U_proxy, U_proxy, NumSolverThreads, drop_tol, BigProblem, ReducedSolve=False)

    imag_part = Mat_Method_Calc_Imag_Part(Array, Integration_Order, Theta0Sol, bilinear_bonus_int_order, fes2, mesh, inout, alpha, 
        np.squeeze(np.asarray(Theta1Sols)), sigma, U_proxy, U_proxy, U_proxy, xivec,  NumSolverThreads, drop_tol, BigProblem, ReducedSolve=False)
    
    for Num in range(len(Array)):
        TensorArray[Num, :] = real_part[Num,:] + N0.flatten()
        TensorArray[Num, :] += 1j * imag_part[Num,:]

        R = TensorArray[Num, :].real.reshape(3, 3)
        I = TensorArray[Num, :].imag.reshape(3, 3)
        EigenValues[Num, :] = np.sort(np.linalg.eigvals(R)) + 1j * np.sort(np.linalg.eigvals(I))
        
        

    print("Frequency Sweep complete")

    # if use_integral is False:
    #     Theta1Sols = np.zeros((ndof2, NumberofFrequencies, 3), dtype=complex)
    #     for i in range(len(Outputs)):
    #         Theta1Sols[:, i, :] = np.asarray(np.squeeze(Outputs[i]))

    #     print(' Computing coefficients')

    #     Core_Distribution = []
    #     Count_Distribution = []
    #     for i in range(CPUs):
    #         Core_Distribution.append([])
    #         Count_Distribution.append([])
    #     # Distribute frequencies between the cores
    #     CoreNumber = 0
    #     for i, Omega in enumerate(Array):
    #         Core_Distribution[CoreNumber].append(Omega)
    #         Count_Distribution[CoreNumber].append(i)
    #         if CoreNumber == CPUs - 1:
    #             CoreNumber = 0
    #         else:
    #             CoreNumber += 1
    #     # Distribute the lower dimensional solutions
    #     Sols = []
    #     for i in range(CPUs):
    #         TempArray = np.zeros([ndof2, len(Count_Distribution[i]), 3], dtype=complex)
    #         for j, Sim in enumerate(Count_Distribution[i]):
    #             TempArray[:, j, :] = Theta1Sols[:, Sim, :]
    #         Sols.append(TempArray)

        # # I'm aware that pre and post multiplying by identity of size ndof2 is slower than using K and A matrices outright,
        # # however this allows us to reuse the Construct_Matrices function rather than add (significantly) more code.
        # identity1 = sp.identity(ndof2)
        # # Cteate the inputs
        # Runlist = []
        # manager = multiprocessing.Manager()
        # counter = manager.Value('i', 0)
        # for i in range(CPUs):
        #     Runlist.append((Core_Distribution[i], mesh, fes, fes2, Sols[i], identity1, identity1, identity1,
        #                     Theta0Sol, xivec, alpha, sigma, mu_inv, inout, N0, NumberofFrequencies, counter,
        #                     False, 0, 0, Order, Integration_Order, bilinear_bonus_int_order, use_integral))

        # # Run on the multiple cores
        # # Edit James Elgy: changed how pool was generated to 'spawn': see
        # # https://britishgeologicalsurvey.github.io/science/python-forking-vs-spawn/
        # with multiprocessing.get_context('spawn').Pool(CPUs) as pool:
        #     Outputs = pool.starmap(Theta1_Lower_Sweep, Runlist)

        # for i, Output in enumerate(Outputs):
        #     for j, Num in enumerate(Count_Distribution[i]):
        #         TensorArray[Num, :] = Output[0][j]
        #         EigenValues[Num, :] = Output[1][j]


    return TensorArray, EigenValues, N0, numelements, (ndof, ndof2)




def init_workers(lock, stream):
    tqdm.tqdm.set_lock(lock)
    sys.stderr = stream
