"""
Edit 06 Aug 2022: James Elgy
Changed how N0 was calculated for PODSweep to be consistent with PODSweepMulti.
Changed pool generation to spawn to fix linux bug.

"""
#Importing
import gc
import os
from contextlib import contextmanager
import sys
import time
import math
import multiprocessing as multiprocessing
multiprocessing.freeze_support()
import warnings
from warnings import warn
from tqdm import tqdm
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
from ..POD.calc_error_certificates import *
from ..Core_MPT.imap_execution import *
from ..Core_MPT.supress_stdout import *
from ..Core_MPT.Solve_Theta_0_Problem import *
from ..Core_MPT.Calculate_N0 import *
from ..Core_MPT.Theta0_Postprocessing import *
from ..Core_MPT.Construct_Matrices import *
from ..POD.Truncated_SVD import *
from ..POD.Constuct_ROM import *
from ..POD.Construct_Linear_System import *

sys.path.insert(0,"Settings")
from Settings import SolverParameters, DefaultSettings, IterativePODParameters

# Importing matplotlib for plotting comparisons
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator



def PODSweepMulti(Object, Order, alpha, inorout, mur, sig, Array, PODArray, PODTol, PlotPod, CPUs, sweepname, SavePOD,
                  PODErrorBars, BigProblem, Integration_Order, Additional_Int_Order, Order_L2,  curve=5, recoverymode=False, NumSolverThreads='default', save_U=False):
    print('Running as parallel POD')

    timing_dictionary = {}

    timing_dictionary['start_time'] = time.time()

    EigenValues, Mu0, N0, NumberofFrequencies, NumberofSnapshots, TensorArray,  inout, mesh, mu_inv, numelements, sigma, bilinear_bonus_int_order = MPT_Preallocation(
        Array, Object, PODArray, curve, inorout, mur, sig, Order, Order_L2, sweepname)
    # Set up the Solver Parameters
    Solver, epsi, Maxsteps, Tolerance, _, use_integral = SolverParameters()

    #########################################################################
    # Theta0
    # This section solves the Theta0 problem to calculate both the inputs for
    # the Theta1 problem and calculate the N0 tensor

    Theta0Sol, Theta0i, Theta0j, fes, ndof, evec = Solve_Theta_0_Problem(Additional_Int_Order, CPUs, Maxsteps, Order, Solver,
                                                                   Tolerance, alpha, epsi, inout, mesh, mu_inv,
                                                                   recoverymode, sweepname)

    if recoverymode is False:
        np.save('Results/' + sweepname + '/Data/Theta0', Theta0Sol)

    Theta0Sol = Theta0_Postprocessing(Additional_Int_Order, Theta0Sol, fes)

    # Calculate the N0 tensor
    N0 = Calculate_N0(Integration_Order, N0, Theta0Sol, Theta0i, Theta0j, alpha, mesh, mu_inv)

    timing_dictionary['Theta0'] = time.time()

    #########################################################################
    # Theta1
    # This section solves the Theta1 problem and saves the solution vectors

    print(' solving theta1 snapshots')
    # Setup the finite element space
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes2 = HCurl(mesh, order=Order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    # Count the number of degrees of freedom
    ndof2 = fes2.ndof

    # Define the vectors for the right hand side
    xivec = [CoefficientFunction((0, -z, y)), CoefficientFunction((z, 0, -x)), CoefficientFunction((-y, x, 0))]

    if recoverymode is False:
        # Work out where to send each frequency
        Theta1_CPUs = min(NumberofSnapshots, multiprocessing.cpu_count(), CPUs)
        Core_Distribution = []
        Count_Distribution = []
        for i in range(Theta1_CPUs):
            Core_Distribution.append([])
            Count_Distribution.append([])

        # Distribute between the cores
        CoreNumber = 0
        count = 1
        for i, Omega in enumerate(PODArray):
            Core_Distribution[CoreNumber].append(Omega)
            Count_Distribution[CoreNumber].append(i)
            if CoreNumber == CPUs - 1 and count == 1:
                count = -1
            elif CoreNumber == 0 and count == -1:
                count = 1
            else:
                CoreNumber += count

        # Create the inputs
        Runlist = []
        manager = multiprocessing.Manager()
        counter = manager.Value('i', 0)

        for i in range(len(PODArray)):
            if PlotPod == True:
                Runlist.append((np.asarray([PODArray[i]]), mesh, fes, fes2, Theta0Sol, xivec, alpha, sigma, mu_inv, inout,
                                Tolerance, Maxsteps, epsi, Solver, N0, NumberofSnapshots, True, True, counter,
                                BigProblem, Order, NumSolverThreads,Integration_Order, Additional_Int_Order, 'Theta1_Sweep'))
            else:
                Runlist.append((np.asarray([PODArray[i]]), mesh, fes, fes2, Theta0Sol, xivec, alpha, sigma, mu_inv, inout,
                                Tolerance, Maxsteps, epsi, Solver, N0, NumberofSnapshots, True, False, counter,
                                BigProblem, Order, NumSolverThreads, Integration_Order, Additional_Int_Order, 'Theta1_Sweep'))

        # Run on the multiple cores
        multiprocessing.freeze_support()
        tqdm.tqdm.set_lock(multiprocessing.RLock())
        if ngsglobals.msg_level != 0:
            to = os.devnull
        else:
            to = os.devnull
        with supress_stdout(to=to):
            with multiprocessing.get_context("spawn").Pool(Theta1_CPUs, maxtasksperchild=1, initializer=tqdm.tqdm.set_lock, initargs=(tqdm.tqdm.get_lock(),)) as pool:
                Outputs = list(tqdm.tqdm(pool.imap(imap_version, Runlist, chunksize=1), total=len(Runlist), desc='Solving Theta1 Snapshots', dynamic_ncols=True,
                                         position=0, leave=True))

        try:
            pool.terminate()
            print('manually closed pool')
        except:
            print('Pool has already closed.')

        # Unpack the results
        if BigProblem == True:
            Theta1Sols = np.zeros([ndof2, NumberofSnapshots, 3], dtype=np.complex64)
        else:
            Theta1Sols = np.zeros([ndof2, NumberofSnapshots, 3], dtype=complex)
        if PlotPod == True:
            PODTensors = np.zeros([NumberofSnapshots, 9], dtype=complex)
            PODEigenValues = np.zeros([NumberofSnapshots, 3], dtype=complex)

        for i in range(len(Outputs)):
            if PlotPod is True:
                PODEigenValues[i, :] = Outputs[i][1][0]
                PODTensors[i, :] = Outputs[i][0][0]
                for j in range(ndof2):
                    Theta1Sols[j,i,:] = Outputs[i][2][j][0]
            else:
                for j in range(ndof2):
                    Theta1Sols[j, i, :] = Outputs[i][0][j][0]

        timing_dictionary['Theta1'] = time.time()

        ########################################################################
        # Create the ROM

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
    if save_U is True and recoverymode is False:
        np.save('Results/' + sweepname + '/Data/U1_truncated', u1Truncated)
        np.save('Results/' + sweepname + '/Data/U2_truncated', u2Truncated)
        np.save('Results/' + sweepname + '/Data/U3_truncated', u3Truncated)
        np.savetxt('Results/' + sweepname + '/Data/PODTensors.csv', PODTensors, delimiter=',')
        np.savetxt('Results/' + sweepname + '/Data/PODEigenvalues.csv', PODEigenValues, delimiter=',')
    ########################################################################
    # Create the ROM

    a0, a1, r1, r2, r3, read_vec, u, v, write_vec = Construct_ROM(Additional_Int_Order, BigProblem, Mu0, Theta0Sol,
                                                                  alpha, epsi, fes, fes2, inout, mu_inv, sigma, xivec)

    if PODErrorBars is True:
        HA0H1, HA0H2, HA0H3, HA1H1, HA1H2, HA1H3, HR1, HR2, HR3, ProL, RerrorReduced1, RerrorReduced2, RerrorReduced3, fes0, ndof0 = Construct_Linear_System(
            PODErrorBars, a0, a1, cutoff, dom_nrs_metal, fes2, mesh, ndof2, r1, r2, r3, read_vec, u1Truncated, u2Truncated,
            u3Truncated, write_vec)
    else:
        HA0H1, HA0H2, HA0H3, HA1H1, HA1H2, HA1H3, HR1, HR2, HR3, _, _, _, _, _, _ = Construct_Linear_System(
            PODErrorBars, a0, a1, cutoff, dom_nrs_metal, fes2, mesh, ndof2, r1, r2, r3, read_vec, u1Truncated, u2Truncated,
            u3Truncated, write_vec)

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

        G_Store = np.zeros([2 * cutoff + 1, 2 * cutoff + 1, 6], dtype=complex)
        G_Store[:, :, 0] = np.transpose(np.conjugate(RerrorReduced1)) @ MR1
        G_Store[:, :, 1] = np.transpose(np.conjugate(RerrorReduced2)) @ MR2
        G_Store[:, :, 2] = np.transpose(np.conjugate(RerrorReduced3)) @ MR3
        G_Store[:, :, 3] = np.transpose(np.conjugate(RerrorReduced1)) @ MR2
        G_Store[:, :, 4] = np.transpose(np.conjugate(RerrorReduced1)) @ MR3
        G_Store[:, :, 5] = np.transpose(np.conjugate(RerrorReduced2)) @ MR3

        # Clear the variables
        RerrorReduced1, RerrorReduced2, RerrorReduced3 = None, None, None
        MR1, MR2, MR3 = None, None, None
        fes0, m, c, inverse = None, None, None, None

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
        # print(f'alphaLB = {alphaLB}')

    else:
        alphaLB, G_Store = False, False

        # Clear the variables
        fes3, amax, apre, pre, invh1, m = None, None, None, None, None, None
    timing_dictionary['ROM'] = time.time()

    ######################################################################
    # Produce the sweep on the lower dimensional space
    g = np.zeros([cutoff, NumberofFrequencies, 3], dtype=complex)
    for k, omega in enumerate(Array):
        g[:, k, 0] = np.linalg.solve(HA0H1 + HA1H1 * omega, HR1 * omega)
        g[:, k, 1] = np.linalg.solve(HA0H2 + HA1H2 * omega, HR2 * omega)
        g[:, k, 2] = np.linalg.solve(HA0H3 + HA1H3 * omega, HR3 * omega)

    # Work out where to send each frequency
    timing_dictionary['SolvedSmallerSystem'] = time.time()
    Tensor_CPUs = min(NumberofFrequencies, multiprocessing.cpu_count(), CPUs)
    Tensor_CPUs = 1

    # g = Theta1Sols

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
        # TempArray = np.zeros([cutoff, len(Count_Distribution[i]), 3], dtype=complex)
        TempArray = np.zeros([cutoff, len(Count_Distribution[i]), 3], dtype=complex)
        for j, Sim in enumerate(Count_Distribution[i]):
            TempArray[:, j, :] = g[:, Sim, :]
        Lower_Sols.append(TempArray)

    timing_dictionary['AssignedCores'] = time.time()


    # Depending on if the user has specified using the slower integral method. This is known to produce the correct
    # answer. Also used if PODErrorBars are required, since it calculates error certificates at the same time as the
    # tensor coefficients.
    use_integral_debug = False
    if use_integral is True or use_integral_debug is True:
        # Cteate the inputs
        Runlist = []
        manager = multiprocessing.Manager()
        counter = manager.Value('i', 0)
        for i in range(Tensor_CPUs):
            Runlist.append((Core_Distribution[i], mesh, fes, fes2, Lower_Sols[i], u1Truncated, u2Truncated, u3Truncated,
                            Theta0Sol, xivec, alpha, sigma, mu_inv, inout, N0, NumberofFrequencies, counter, PODErrorBars,
                            alphaLB, G_Store, Order, Integration_Order, Additional_Int_Order, use_integral))

        # Run on the multiple cores
        # Edit James Elgy: changed how pool was generated to 'spawn': see
        # https://britishgeologicalsurvey.github.io/science/python-forking-vs-spawn/
        with multiprocessing.get_context('spawn').Pool(Tensor_CPUs) as pool:
            Outputs = pool.starmap(Theta1_Lower_Sweep, Runlist)

    else:

        At0_array, EU_array_conj, Q_array, T_array, UAt0U_array, UAt0_conj, UH_array, c1_array, c5_array, c7, c8_array = Construct_Matrices(
            Integration_Order, Theta0Sol, bilinear_bonus_int_order, fes2, inout, mesh, mu_inv, sigma, sweepname, u,
            u1Truncated, u2Truncated, u3Truncated, v, xivec)

        timing_dictionary['BuildSystemMatrices'] = time.time()

        runlist = []
        for i in range(Tensor_CPUs):
            runlist.append((Core_Distribution[i], Q_array, c1_array, c5_array, c7, c8_array, At0_array, UAt0_conj,
                            UAt0U_array, T_array, EU_array_conj, UH_array, Lower_Sols[i], G_Store, cutoff, fes2.ndof,
                            alpha, False))

        with multiprocessing.get_context('spawn').Pool(Tensor_CPUs) as pool:
            Outputs = pool.starmap(Theta1_Lower_Sweep_Mat_Method, runlist)

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

    else:
        for i, Output in enumerate(Outputs):
            for j, Num in enumerate(Count_Distribution[i]):
                if PODErrorBars == True:
                    TensorArray[Num, :] = Output[0][j]
                    TensorArray[Num, :] = Output[0][j] + N0.flatten()
                    R = TensorArray[Num, :].real.reshape(3, 3)
                    I = TensorArray[Num, :].imag.reshape(3, 3)
                    EigenValues[Num, :] = np.sort(np.linalg.eigvals(R)) + 1j * np.sort(np.linalg.eigvals(I))
                    # ErrorTensors[Num, :] = Output[2][j]
                else:
                    TensorArray[Num, :] = Output[0][j] + N0.flatten()
                    R = TensorArray[Num, :].real.reshape(3, 3)
                    I = TensorArray[Num, :].imag.reshape(3, 3)
                    EigenValues[Num, :] = np.sort(np.linalg.eigvals(R)) + 1j * np.sort(np.linalg.eigvals(I))

    print(' reduced order systems solved')

    if (use_integral is False) and (use_integral_debug is False) and (PODErrorBars is True):
        print(' Computing Errors')
        # For parallelisation, this has to be a separate function. Also with the intention that we can also reuse this
        # function in the other POD functions. I.e. PODSweep, PODSweepMulti.
        ErrorTensors = np.zeros((len(Array), 6))
        for i in range(Tensor_CPUs):
            Distributed_Errors = calc_error_certificates(Core_Distribution[i], alphaLB, G_Store, cutoff, alpha, Lower_Sols[i])
            ErrorTensors[Count_Distribution[i],:] = Distributed_Errors

    print(' frequency sweep complete')
    timing_dictionary['Tensors'] = time.time()
    np.save('Results/' + sweepname + f'/Data/Timings_cpus={CPUs}.npy', timing_dictionary)

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







