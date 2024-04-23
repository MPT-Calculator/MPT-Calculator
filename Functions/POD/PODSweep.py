"""
Edit 06 Aug 2022: James Elgy
Changed how N0 was calculated for PODSweep to be consistent with PODSweepMulti.
Changed pool generation to spawn to fix linux bug.

"""

"""
Paul Ledger (2024 edit)
Added drop_tol to reduce memory usage when building full matrices including interiors
"""
#Importing

import os
import sys
import time
import math
import multiprocessing as multiprocessing
import warnings
from warnings import warn
import tqdm
import cmath
import numpy as np
import scipy.signal
import scipy.sparse as sp
import scipy.sparse.linalg as spl

import netgen.meshing as ngmeshing
from ngsolve import *

sys.path.insert(0,"Functions")
from ..Core_MPT.Theta0 import *
from ..Core_MPT.Theta1 import *
from ..Core_MPT.Theta1_Sweep import *
from ..Core_MPT.Theta1_Lower_Sweep import *
from ..Core_MPT.Theta1_Lower_Sweep_Mat_Method import *
from ..Core_MPT.MPT_Preallocation import *
from ..Core_MPT.Solve_Theta_0_Problem import *
from ..Core_MPT.Calculate_N0 import *
from ..Core_MPT.Theta0_Postprocessing import *
from ..Core_MPT.Construct_Matrices import *
from ..POD.Truncated_SVD import *
# from ..POD.Construct_Linear_System import *
# from ..POD.Constuct_ROM import *
from ..POD.calc_error_certificates import *
from ..POD.Construct_Linear_System import *
from ..Core_MPT.Mat_Method_Calc_Real_Part import *
from ..Core_MPT.Mat_Method_Calc_Imag_Part import *


sys.path.insert(0,"Settings")
from Settings import SolverParameters, DefaultSettings, IterativePODParameters

# Importing matplotlib for plotting comparisons
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import time


def PODSweep(Object, Order, alpha, inorout, mur, sig, Array, PODArray, PODTol, PlotPod, sweepname, SavePOD,
             PODErrorBars, BigProblem, NumSolverThreads, Integration_Order, Additional_Int_Order, Order_L2, drop_tol, curve=5,
             recoverymode=False, save_U=False):

    print(' Running as POD')

    EigenValues, Mu0, N0, NumberofFrequencies, NumberofSnapshots, TensorArray,  inout, mesh, mu_inv, numelements, sigma, bilinear_bonus_int_ord = MPT_Preallocation(
        Array, Object, PODArray, curve, inorout, mur, sig, Order, Order_L2, sweepname, NumSolverThreads, drop_tol)

    Solver, epsi, Maxsteps, Tolerance, _, use_integral = SolverParameters()

    #########################################################################
    # Theta0
    # This section solves the Theta0 problem to calculate both the inputs for
    # the Theta1 problem and calculate the N0 tensor

    # Here, we either load theta0 or calculate.
    Theta0Sol, Theta0i, Theta0j, fes, ndof, evec = Solve_Theta_0_Problem(Additional_Int_Order, 1, Maxsteps, Order,
                                                                         Solver,
                                                                         Tolerance, alpha, epsi, inout, mesh, mu_inv,
                                                                         recoverymode, sweepname)
    if recoverymode is False:
        np.save('Results/' + sweepname + '/Data/Theta0', Theta0Sol)

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
    ndof2 = fes2.ndof

    # Define the vectors for the right hand side
    xivec = [CoefficientFunction((0, -z, y)), CoefficientFunction((z, 0, -x)), CoefficientFunction((-y, x, 0))]

    if recoverymode is False:
        if BigProblem == True:
            Theta1Sols = np.zeros([ndof2, NumberofSnapshots, 3], dtype=np.complex64)
        else:
            Theta1Sols = np.zeros([ndof2, NumberofSnapshots, 3], dtype=complex)

        if (PlotPod is False) or (use_integral is False): 
            ComputeTensors = False
        else:
            ComputeTensors = True
        
        
        if ComputeTensors == True:
            PODTensors, PODEigenValues, Theta1Sols[:, :, :] = Theta1_Sweep(PODArray, mesh, fes, fes2, Theta0Sol, xivec,
                                                                           alpha, sigma, mu_inv, inout, Tolerance, Maxsteps,
                                                                           epsi, Solver, N0, NumberofFrequencies, True,
                                                                           True, False, BigProblem, Order, NumSolverThreads, Integration_Order, 
                                                                           Additional_Int_Order, bilinear_bonus_int_ord, drop_tol)
        else:
            Theta1Sols[:, :, :] = Theta1_Sweep(PODArray, mesh, fes, fes2, Theta0Sol, xivec, alpha, sigma, mu_inv, inout,
                                               Tolerance, Maxsteps, epsi, Solver, N0, NumberofFrequencies, True, False,
                                               False, BigProblem, Order, NumSolverThreads, Integration_Order, Additional_Int_Order,
                                               bilinear_bonus_int_ord, drop_tol)
        print(' solved theta1 problems     ')
        
        
        if use_integral is False and PlotPod is True:
            
            PODTensors = np.zeros([NumberofSnapshots, 9], dtype=complex)
            PODEigenValues = np.zeros([NumberofSnapshots, 3], dtype=complex)
            U_proxy = sp.eye(fes2.ndof)

            real_part = Mat_Method_Calc_Real_Part(bilinear_bonus_int_ord, fes2, inout, mu_inv, alpha, np.squeeze(np.asarray(Theta1Sols)),
                U_proxy, U_proxy, U_proxy, NumSolverThreads, drop_tol, BigProblem, ReducedSolve=False)

            imag_part = Mat_Method_Calc_Imag_Part(PODArray, Integration_Order, Theta0Sol, bilinear_bonus_int_ord, fes2, mesh, inout, alpha, np.squeeze(np.asarray(Theta1Sols)),
                sigma, U_proxy, U_proxy, U_proxy, xivec,  NumSolverThreads, drop_tol, BigProblem, ReducedSolve=False)
            
            for Num in range(len(PODArray)):
                PODTensors[Num, :] = real_part[Num,:] + N0.flatten()
                PODTensors[Num, :] += 1j * imag_part[Num,:]

                R = PODTensors[Num, :].real.reshape(3, 3)
                I = PODTensors[Num, :].imag.reshape(3, 3)
                PODEigenValues[Num, :] = np.sort(np.linalg.eigvals(R)) + 1j * np.sort(np.linalg.eigvals(I))
    
    
            
        
        #########################################################################
        # POD

        cutoff, u1Truncated, u2Truncated, u3Truncated = Truncated_SVD(NumberofSnapshots, PODTol, Theta1Sols)
        plt.savefig('Results/' + sweepname + '/Graphs/SVD_Decay.pdf')

    else:
        print(' Loading truncated vectors')
        # Loading in Left Singular Vectors:
        u1Truncated = np.load('Results/' + sweepname + '/Data/U1_truncated.npy')
        u2Truncated = np.load('Results/' + sweepname + '/Data/U2_truncated.npy')
        u3Truncated = np.load('Results/' + sweepname + '/Data/U3_truncated.npy')
        try:
            PODTensors = np.genfromtxt('Results/' + sweepname + '/Data/PODTensors.csv', dtype=complex, delimiter=',')
            PODEigenValues = np.genfromtxt('Results/' + sweepname + '/Data/PODEigenvalues.csv', dtype=complex,
                                           delimiter=',')
        except FileNotFoundError:
            print('PODTensors.csv or PODEigenValues.csv not found. Continuing with POD tensor coefficients set to 0')
            PODTensors = np.zeros((len(PODArray), 9), dtype=complex)
            PODEigenValues = np.zeros((len(PODArray), 3), dtype=complex)

        cutoff = u1Truncated.shape[1]
        print(' Loaded Data')

    # save_U = True
    if save_U is True:
        np.save('Results/' + sweepname + '/Data/U1_truncated', u1Truncated)
        np.save('Results/' + sweepname + '/Data/U2_truncated', u2Truncated)
        np.save('Results/' + sweepname + '/Data/U3_truncated', u3Truncated)
        np.savetxt('Results/' + sweepname + '/Data/PODTensors.csv', PODTensors, delimiter=',')
        np.savetxt('Results/' + sweepname + '/Data/PODEigenvalues.csv', PODEigenValues, delimiter=',')
    ########################################################################
    # Create the ROM
    
    if PODErrorBars is True:
        HA0H1, HA0H2, HA0H3, HA1H1, HA1H2, HA1H3, HR1, HR2, HR3, ProL, RerrorReduced1, RerrorReduced2, RerrorReduced3, fes0, ndof0 = Construct_Linear_System(Additional_Int_Order, BigProblem, Mu0, Theta0Sol, alpha, epsi, fes, fes2, inout, mu_inv, sigma,
                  xivec, NumSolverThreads, drop_tol, u1Truncated, u2Truncated, u3Truncated, dom_nrs_metal, PODErrorBars)
    else:
        HA0H1, HA0H2, HA0H3, HA1H1, HA1H2, HA1H3, HR1, HR2, HR3, _, _, _, _, _, _ = Construct_Linear_System(Additional_Int_Order, BigProblem, Mu0, Theta0Sol, alpha, epsi, fes, fes2, inout, mu_inv, sigma,
                  xivec, NumSolverThreads, drop_tol, u1Truncated, u2Truncated, u3Truncated, dom_nrs_metal, PODErrorBars)
        
    # Clear the variables
    A0H, A1H = None, None
    a0, a1 = None, None

    ########################################################################
    # Sort out the error bounds
    if PODErrorBars == True:
        if BigProblem == True:
            MR1 = np.zeros([ndof0, cutoff * 2 + 1], dtype=np.complex64)
            MR2 = np.zeros([ndof0, cutoff * 2 + 1], dtype=np.complex64)
            MR3 = np.zeros([ndof0, cutoff * 2 + 1], dtype=np.complex64)
        else:
            MR1 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)
            MR2 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)
            MR3 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)

        u, v = fes0.TnT()

        m = BilinearForm(fes0)
        m += SymbolicBFI(InnerProduct(u, v), bonus_intorder=Additional_Int_Order)
        f = LinearForm(fes0)
        m.Assemble()
        c = Preconditioner(m, "local")
        c.Update()
        inverse = CGSolver(m.mat, c.mat, precision=1e-20, maxsteps=500)

        ErrorGFU = GridFunction(fes0)
        for i in range(2 * cutoff + 1):
            # E1
            ProL.vec.data.FV().NumPy()[:] = RerrorReduced1[:, i]
            ProL.vec.data -= m.mat * ErrorGFU.vec
            ErrorGFU.vec.data += inverse * ProL.vec
            MR1[:, i] = ErrorGFU.vec.FV().NumPy()

            # E2
            ProL.vec.data.FV().NumPy()[:] = RerrorReduced2[:, i]
            ProL.vec.data -= m.mat * ErrorGFU.vec
            ErrorGFU.vec.data += inverse * ProL.vec
            MR2[:, i] = ErrorGFU.vec.FV().NumPy()

            # E3
            ProL.vec.data.FV().NumPy()[:] = RerrorReduced3[:, i]
            ProL.vec.data -= m.mat * ErrorGFU.vec
            ErrorGFU.vec.data += inverse * ProL.vec
            MR3[:, i] = ErrorGFU.vec.FV().NumPy()

        fes3 = HCurl(mesh, order=Order, dirichlet="outer", gradientdomains=dom_nrs_metal)
        ndof3 = fes3.ndof
        Omega = Array[0]
        u, v = fes3.TnT()
        amax = BilinearForm(fes3)
        amax += SymbolicBFI((mu_inv) * curl(u) * curl(v), bonus_intorder=Additional_Int_Order)
        amax += SymbolicBFI((1 - inout) * epsi * u * v, bonus_intorder=Additional_Int_Order)
        amax += SymbolicBFI(inout * sigma * (alpha ** 2) * Mu0 * Omega * u * v, bonus_intorder=Additional_Int_Order)

        m = BilinearForm(fes3)
        m += SymbolicBFI(u * v, bonus_intorder=Additional_Int_Order)

        apre = BilinearForm(fes3)
        apre += SymbolicBFI(curl(u) * curl(v), bonus_intorder=Additional_Int_Order)
        apre += SymbolicBFI(u * v, bonus_intorder=Additional_Int_Order)
        pre = Preconditioner(amax, "bddc")

        with TaskManager():
            amax.Assemble()
            m.Assemble()
            apre.Assemble()

            # build gradient matrix as sparse matrix (and corresponding scalar FESpace)
            gradmat, fesh1 = fes3.CreateGradient()
            gradmattrans = gradmat.CreateTranspose()  # transpose sparse matrix
            math1 = gradmattrans @ m.mat @ gradmat  # multiply matrices
            math1[0, 0] += 1  # fix the 1-dim kernel
            invh1 = math1.Inverse(inverse="sparsecholesky")

            # build the Poisson projector with operator Algebra:
            proj = IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ m.mat
            projpre = proj @ pre.mat
            evals, evecs = solvers.PINVIT(amax.mat, m.mat, pre=projpre, num=1, maxit=50, printrates=False)

        alphaLB = evals[0]

    else:
        alphaLB, G_Store = False, False

        # Clear the variables
        fes3, amax, apre, pre, invh1, m = None, None, None, None, None, None

    ######################################################################
    # Produce the sweep on the lower dimensional space
    g = np.zeros([cutoff, NumberofFrequencies, 3], dtype=complex)
    for k, omega in enumerate(Array):
        g[:, k, 0] = np.linalg.solve(HA0H1 + HA1H1 * omega, HR1 * omega)
        g[:, k, 1] = np.linalg.solve(HA0H2 + HA1H2 * omega, HR2 * omega)
        g[:, k, 2] = np.linalg.solve(HA0H3 + HA1H3 * omega, HR3 * omega)

    if PODErrorBars is True:
        G_Store = np.zeros([2 * cutoff + 1, 2 * cutoff + 1, 6], dtype=complex)
        G_Store[:, :, 0] = np.transpose(np.conjugate(RerrorReduced1)) @ MR1
        G_Store[:, :, 1] = np.transpose(np.conjugate(RerrorReduced2)) @ MR2
        G_Store[:, :, 2] = np.transpose(np.conjugate(RerrorReduced3)) @ MR3
        G_Store[:, :, 3] = np.transpose(np.conjugate(RerrorReduced1)) @ MR2
        G_Store[:, :, 4] = np.transpose(np.conjugate(RerrorReduced1)) @ MR3
        G_Store[:, :, 5] = np.transpose(np.conjugate(RerrorReduced2)) @ MR3
    else:
        G_Store = False
        alphaLB = False

    Tensor_CPUs = 1

    Core_Distribution = []
    Count_Distribution = []
    for i in range(Tensor_CPUs):
        Core_Distribution.append([])
        Count_Distribution.append([])
    # Distribute frequencies between the cores
    CoreNumber = 0
    for i, Omega in enumerate(Array):
        Core_Distribution[CoreNumber].append(Omega)
        Count_Distribution[CoreNumber].append(i)
        if CoreNumber == Tensor_CPUs - 1:
            CoreNumber = 0
        else:
            CoreNumber += 1
    # Distribute the lower dimensional solutions
    Lower_Sols = []
    for i in range(Tensor_CPUs):
        TempArray = np.zeros([cutoff, len(Count_Distribution[i]), 3], dtype=complex)
        for j, Sim in enumerate(Count_Distribution[i]):
            TempArray[:, j, :] = g[:, Sim, :]
        Lower_Sols.append(TempArray)


    # Depending on if the user has specified using the slower integral method. This is known to produce the correct
    # answer. Also used if PODErrorBars are required, since it calculates error certificates at the same time as the
    # tensor coefficients.
    use_integral_debug = False
    Tensor_CPUs = 1 # setting for non-parallel run

    if use_integral is True or use_integral_debug is True:
        # Cteate the inputs
        Runlist = []
        manager = multiprocessing.Manager()
        counter = manager.Value('i', 0)
        for i in range(Tensor_CPUs):
            Runlist.append(
                (Core_Distribution[i], mesh, fes, fes2, Lower_Sols[i], u1Truncated, u2Truncated, u3Truncated,
                 Theta0Sol, xivec, alpha, sigma, _inv, inout, N0, NumberofFrequencies, counter, PODErrorBars,
                 alphaLB, G_Store, Order, Integration_Order, Additional_Int_Order, use_integral))

        # Run on the multiple cores
        # Edit James Elgy: changed how pool was generated to 'spawn': see
        # https://britishgeologicalsurvey.github.io/science/python-forking-vs-spawn/
        with multiprocessing.get_context('spawn').Pool(Tensor_CPUs) as pool:
            Outputs = pool.starmap(Theta1_Lower_Sweep, Runlist)

    else:

        real_part = Mat_Method_Calc_Real_Part(bilinear_bonus_int_ord, fes2, inout, mu_inv, alpha, np.squeeze(np.asarray(Lower_Sols)),
            u1Truncated, u2Truncated, u3Truncated, NumSolverThreads, drop_tol, BigProblem, ReducedSolve=True)

        imag_part = Mat_Method_Calc_Imag_Part(Array, Integration_Order, Theta0Sol, bilinear_bonus_int_ord, fes2, mesh, inout, alpha, np.squeeze(np.asarray(Lower_Sols)),
            sigma, u1Truncated, u2Truncated, u3Truncated, xivec,  NumSolverThreads, drop_tol, BigProblem, ReducedSolve=True)
        
        for Num in range(len(Array)):
            TensorArray[Num, :] = real_part[Num,:] + N0.flatten()
            TensorArray[Num, :] += 1j * imag_part[Num,:]

            R = TensorArray[Num, :].real.reshape(3, 3)
            I = TensorArray[Num, :].imag.reshape(3, 3)
            EigenValues[Num, :] = np.sort(np.linalg.eigvals(R)) + 1j * np.sort(np.linalg.eigvals(I))

    try:
        pool.terminate()
        print('manually closed pool')
    except:
        print('Pool has already closed.')

    # Unpack the outputs
    if use_integral is True or use_integral_debug is True:
        if PODErrorBars == True:
            ErrorTensors = np.zeros([NumberofFrequencies, 6])
        for i, Output in enumerate(Outputs):
            for j, Num in enumerate(Count_Distribution[i]):
                if PODErrorBars == True:
                    TensorArray[Num, :] = Output[0][j]
                    EigenValues[Num, :] = Output[1][j]
                    ErrorTensors[Num, :] = Output[2][j]
                else:
                    TensorArray[Num, :] = Output[0][j]
                    EigenValues[Num, :] = Output[1][j]


    print(' reduced order systems solved')

    if (use_integral is False) and (use_integral_debug is False) and (PODErrorBars is True):
            print(' Computing Errors')
            # For parallelisation, this has to be a separate function. Also with the intention that we can also reuse this
            # function in the other POD functions. I.e. PODSweep, PODSweepMulti.
            ErrorTensors = np.zeros((len(Array), 6))
            for i in range(Tensor_CPUs):
                Distributed_Errors = calc_error_certificates(Core_Distribution[i], alphaLB, G_Store, cutoff, alpha,
                                                             Lower_Sols[i])
                ErrorTensors[Count_Distribution[i], :] = Distributed_Errors

    print(' reduced order systems solved        ')
    print(' frequency sweep complete')

    if PlotPod == True:
        if PODErrorBars == True:
            return TensorArray, EigenValues, N0, PODTensors, PODEigenValues, numelements, ErrorTensors, (ndof, ndof2)
        else:
            return TensorArray, EigenValues, N0, PODTensors, PODEigenValues, numelements, (ndof, ndof2)
    else:
        if PODErrorBars == True:
            return TensorArray, EigenValues, N0, numelements, ErrorTensors, (ndof, ndof2)
        else:
            return TensorArray, EigenValues, N0, numelements, (ndof, ndof2)




