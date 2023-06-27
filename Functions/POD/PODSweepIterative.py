
"""
Edit 06 Aug 2022: James Elgy
Changed how N0 was calculated for PODSweep to be consistent with PODSweepMulti.
Changed pool generation to spawn to fix linux bug.

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
from ..POD.calc_error_certificates import *
from ..Core_MPT.imap_execution import *
from ..Core_MPT.supress_stdout import *
sys.path.insert(0,"Settings")
from Settings import SolverParameters, DefaultSettings, IterativePODParameters, AdditionalOutputs

# Importing matplotlib for plotting comparisons
import matplotlib
# matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from ..Core_MPT.Solve_Theta_0_Problem import *
from ..Core_MPT.Calculate_N0 import *
from ..Core_MPT.Theta0_Postprocessing import *
from ..Core_MPT.Construct_Matrices import *
from ..POD.Truncated_SVD import *
from ..POD.Constuct_ROM import *
from ..POD.Construct_Linear_System import *
from ..Core_MPT.MPT_Preallocation import *
from ..POD.calc_error_certificates import *

# from ngsolve import ngsglobals
# ngsglobals.msg_level = 3


def PODSweepIterative(Object, Order, alpha, inorout, mur, sig, Array, PODArray, PlotPod, sweepname, SavePOD,
             PODErrorBars, BigProblem, Integration_Order, Additional_Int_Order, curve=5,  use_parallel=False, cpus='default', save_U=False):
    """
    James Elgy - 2022
    Iterative version of the existing POD method where additional snapshots are placed in regions of high uncertainty.
    The function works by first calculating the original logarithmically spaced distribution using PODArray and
    calculating error certificates for each frequency in Array. Scipy FindPeaks is then used to calculate the most
    effective frequency to place an additional snapshot and a new theta1 snapshot solution is computed. This new theta1
    solution and its corresponding frequency is then appended to the end of the original Theta1Sols and PODArray list.
    The ROM is then recomputed and new error certificates are calculated. This repeats.
    Parameters
    ----------
    Object: (str) .vol file name to the object in question.
    Order: (int) Order of finite element approximation
    alpha: (float) alpha scaling term to scale unit object
    inorout: (dict) dictionary of the form {'obj1': 1, 'obj2': 1, 'air':0} to flag regions inside the object.
    mur: (dict) dictionary of mur for each different region in the mesh.
    sig: (dict) dictionary of conductivities for each different region in the mesh.
    Array: (list/numpyarray) list of final evaluation frequencies to consider.
    PODArray: (list/numpyarray) list of initial logarithmically spaced POD frequencies
    PODTol: (float) POD SVD tructation tolerance.
    PlotPod: (bool) option to plot the POD output.
    sweepname: (str) name of object results path. Used for saving.
    SavePOD: (bool) option to save left singular vector and theta0 to disk.
    PODErrorBars: (bool) flag to calculate POD errorbars. For this function, should be set to True.
    BigProblem: (bool) option to reduce the floating point percision to single. Saves memory.
    tol: (float) CGSolver tolerance.
    curve=5: (int) order of polynomial approximation for curved surface elements
    prism_flag=False: (bool) option for if mesh contains prismatic elements. Will adjust integration order when
                      calculating POD tensors.
    use_parallel=False: (bool) option to run through using a parallel implementation.
    use_parallel=False: (bool) option o run through using a parallel implementation.

    Returns
    -------
    TensorArray: (numpyarray) Nx9 array of complex tensor coefficients stored in a row major format.
    EigenValues: (numpyarray) Nx3 array of complex eigenvalues (eig(R)+1j*eig(I)) sorted in assending order.
    N0: (numpyarray) 3x3 real coefficinets of N0
    numelements: (int) number of elements used in the discretisation.
    PODTensors: (numpyarray) n'x9 array of complex tensor coefficinets corresponding to the snapshot frequencies.
    ErrorTensors: (numpyarray) Nx6 array containing the lower triangular part for the error certificates. Stored as
    [e_11, e_22, e_33, e_12, e_13, e_23].
    (ndof, ndof2): (tuple) number of degrees of freedom for the theta0 and theta1 problems.
    PODArray: (numpyarray) updated array containing new POD frequencies
    PODArray_orig: (numpyarray) original POD distribution.
    TensorArray_orig: (numpyarray) Nx9 array for original tensor coefficients computed using the original POD snapshots.
    EigenValues_orig: (numpyarray) Nx3 array for original eigenvalues computed using the original POD snapshots
    ErrorTensors_orig: (numpyarray) Nx6 array for error certificates computed using the original POD snapshots
    PODEigenValues_orig: (numpyarray) nx3 array of eigenvalues corresponding to the original snapshot distribution.
    PODTensors_orig: (numpyarray) Nx9 array of tensor coefficients for the original snapshot distribution.
    """


    timing_dictionary = {}
    timing_dictionary['StartTime'] = time.time()

    print('Running iterative POD')
    print(f'Parallel Mode? : {use_parallel}')

    Object = Object[:-4] + ".vol"
    # Set up the Solver Parameters
    Solver, epsi, Maxsteps, Tolerance, _, use_integral = SolverParameters()
    CPUs,BigProblem,PODPoints,PODTol,OldMesh, OldPOD, NumSolverThreads = DefaultSettings()
    _, PODErrorBars, _, _, _, _ = AdditionalOutputs()

    CPUs = cpus

    EigenValues, Mu0, N0, NumberofFrequencies, NumberofSnapshots, TensorArray,  inout, mesh, mu_inv, numelements, sigma, bilinear_bonus_int_order = MPT_Preallocation(
        Array, Object, PODArray, curve, inorout, mur, sig, Order, 0, sweepname)

    # Updating PODErrorBars so that the function will always compute error certificates. We do it here so that the user
    # has the option not to show them in the final output plots.
    if PODErrorBars == False:
        PODErrorBars = True

    # Updating Array so that it doesn't contain the endpoints
    # Array = np.logspace(2,7,40)

    # Setting XLim for plots to be extent of original Array:
    x_min = Array[0]
    x_max = Array[-1]
    # Array_orig = Array

    # Adding POD snapshots to Array. This is so that we can check that error goes to 0 at snapshots.
    Array = np.append(Array, PODArray)
    Array = np.sort(np.unique(Array))
    NumberofFrequencies = len(Array)
    Array_orig = Array

    mask = np.zeros(len(Array))
    for i in range(len(mask)):
        if Array[i] not in PODArray:
            mask[i] = 1

    #########################################################################
    # Theta0
    # This section solves the Theta0 problem to calculate both the inputs for
    # the Theta1 problem and calculate the N0 tensor

    # Setup the finite element space

    # To enable matrix multiplication with consistent sizes, the gradient domains are introduced for theta0
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes = HCurl(mesh, order=Order, dirichlet="outer", gradientdomains=dom_nrs_metal)
    # fes = HCurl(mesh, order=Order, dirichlet="outer", flags = { "nograds" : True })
    # Count the number of degrees of freedom
    ndof = fes.ndof

    # Define the vectors for the right hand side
    evec = [CoefficientFunction((1, 0, 0)), CoefficientFunction((0, 1, 0)), CoefficientFunction((0, 0, 1))]



    ### SOLVING FOR THETA_0:
    # Theta0
    # This section solves the Theta0 problem to calculate both the inputs for
    # the Theta1 problem and calculate the N0 tensor

    Theta0Sol, Theta0i, Theta0j, fes, ndof, evec = Solve_Theta_0_Problem(Additional_Int_Order, CPUs, Maxsteps, Order, Solver,
                                                                   Tolerance, alpha, epsi, inout, mesh, mu_inv,
                                                                   False, sweepname)

    np.save('Results/' + sweepname + '/Data/Theta0', Theta0Sol)

    Theta0Sol = Theta0_Postprocessing(Additional_Int_Order, Theta0Sol, fes)

    # Calculate the N0 tensor
    N0 = Calculate_N0(Integration_Order, N0, Theta0Sol, Theta0i, Theta0j, alpha, mesh, mu_inv)



    #########################################################################
    # Theta1
    # This section solves the Theta1 problem to calculate the solution vectors
    # of the snapshots. In the iterative POD method, theta1 solutions must be avaliable so that a new solution can be
    # appended. Since these are not saved to disk (too large) they are recalculated.

    # Setup the finite element space
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes2 = HCurl(mesh, order=Order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    ndof2 = fes2.ndof

    # Define the vectors for the right hand side
    xivec = [CoefficientFunction((0, -z, y)), CoefficientFunction((z, 0, -x)), CoefficientFunction((-y, x, 0))]

    if BigProblem == True:
        Theta1Sols = np.zeros([ndof2, NumberofSnapshots, 3], dtype=np.complex64)
    else:
        Theta1Sols = np.zeros([ndof2, NumberofSnapshots, 3], dtype=complex)


    if use_parallel is False:
        if PlotPod == True:
            PODTensors, PODEigenValues, Theta1Sols[:, :, :] = Theta1_Sweep(PODArray, mesh, fes, fes2, Theta0Sol, xivec,
                                                                           alpha, sigma, mu_inv, inout, Tolerance, Maxsteps,
                                                                           epsi, Solver, N0, NumberofFrequencies, True,
                                                                           True, False, BigProblem, Order, NumSolverThreads, Integration_Order, Additional_Int_Order)
        else:
            Theta1Sols[:, :, :] = Theta1_Sweep(PODArray, mesh, fes, fes2, Theta0Sol, xivec, alpha, sigma, mu_inv, inout,
                                               Tolerance, Maxsteps, epsi, Solver, N0, NumberofFrequencies, True, False,
                                               False, BigProblem, Order, NumSolverThreads, Integration_Order, Additional_Int_Order)
    else:
        #Work out where to send each frequency
        Theta1_CPUs = min(NumberofSnapshots,multiprocessing.cpu_count(),CPUs)
        Core_Distribution = []
        Count_Distribution = []
        for i in range(Theta1_CPUs):
            Core_Distribution.append([])
            Count_Distribution.append([])


        #Distribute between the cores
        CoreNumber = 0
        count = 1
        for i,Omega in enumerate(PODArray):
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
        for i in range(len(PODArray)):
            if PlotPod == True:
                Runlist.append((np.asarray([PODArray[i]]),mesh,fes,fes2,Theta0Sol,xivec,alpha,sigma,mu_inv,inout,Tolerance,Maxsteps,epsi,Solver,N0,NumberofSnapshots,True,True,counter,BigProblem, Order, NumSolverThreads, Integration_Order, Additional_Int_Order, 'Theta1_Sweep'))
            else:
                Runlist.append((np.asarray([PODArray[i]]),mesh,fes,fes2,Theta0Sol,xivec,alpha,sigma,mu_inv,inout,Tolerance,Maxsteps,epsi,Solver,N0,NumberofSnapshots,True,False,counter,BigProblem, Order, NumSolverThreads, Integration_Order, Additional_Int_Order, 'Theta1_Sweep'))

        #Run on the multiple cores
        multiprocessing.freeze_support()
        tqdm.tqdm.set_lock(multiprocessing.RLock())
        if ngsglobals.msg_level != 0:
            to = sys.stdout
        else:
            to = os.devnull

        print('Computing Theta1')
        with supress_stdout(to=to):
            with multiprocessing.get_context("spawn").Pool(Theta1_CPUs, maxtasksperchild=1, initializer=tqdm.tqdm.set_lock, initargs=(tqdm.tqdm.get_lock(),)) as pool:
                    Outputs = list(tqdm.tqdm(pool.imap(imap_version, Runlist), total=len(Runlist), desc='Solving Theta1 Snapshots',dynamic_ncols=True, position=0, leave=True))

        try:
            pool.terminate()
            print('manually closed pool')
        except:
            print('Pool has already closed.')

        if BigProblem == True:
            Theta1Sols = np.zeros([ndof2, NumberofSnapshots, 3], dtype=np.complex64)
        else:
            Theta1Sols = np.zeros([ndof2, NumberofSnapshots, 3], dtype=complex)
        if PlotPod == True:
            PODTensors = np.zeros([NumberofSnapshots, 9], dtype=complex)
            PODEigenValues = np.zeros([NumberofSnapshots, 3], dtype=complex)

        # print(Theta1Sols.shape)
        # print(np.asarray(Outputs).shape)
        for i in range(len(Outputs)):
            PODEigenValues[i, :] = Outputs[i][1][0]
            PODTensors[i, :] = Outputs[i][0][0]
            for j in range(ndof2):
                Theta1Sols[j,i,:] = Outputs[i][2][j][0]


    print(' solved theta1 problems     ')
    del Outputs
    timing_dictionary['Theta1'] = time.time()


    #########################################################################
    # POD

    fes3 = HCurl(mesh, order=Order, dirichlet="outer", gradientdomains=dom_nrs_metal)
    ndof3 = fes3.ndof
    Omega = Array[0]
    u, v = fes3.TnT()
    amax = BilinearForm(fes3)
    amax += SymbolicBFI((mu_inv ** (-1)) * curl(u) * curl(v), bonus_intorder=Additional_Int_Order)
    amax += SymbolicBFI((1 - inout) * epsi * u * v, bonus_intorder=Additional_Int_Order)
    amax += SymbolicBFI(inout * sigma * (alpha ** 2) * Mu0 * Omega * u * v, bonus_intorder=Additional_Int_Order)

    m = BilinearForm(fes3)
    m += u * v * dx

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
    # print(f'Lower bound alphaLB = {alphaLB} \n')
    # Clear the variables
    fes3, amax, apre, pre, invh1, m = None, None, None, None, None, None

    # Performing iterative POD:
    Max_Error = [np.inf]
    Error_Array = []
    N_Snaps = []
    iter = 0

    N_snaps_per_iter, max_iter, tol, PlotUpdatedPOD = IterativePODParameters()

    Object_Volume = alpha**3 * Integrate(inout, mesh, order=Integration_Order)
    Omega_Max = []
    while Max_Error[-1] / Object_Volume > tol:
        NumberofSnapshots = len(PODArray)
        iter += 1
        if iter > max_iter:
            warnings.warn(f'Iterative POD did not reach set tolerance within {max_iter} iterations')
            break

        print(f' Iteration {iter}')
        EvaluateAtSnapshots = True
        if EvaluateAtSnapshots == True:
            # Updating Array to include new snapshot frequencies.
            NumberofFrequencies = len(Array)
            TensorArray = np.zeros([NumberofFrequencies, 9], dtype=complex)
            EigenValues = np.zeros([NumberofFrequencies, 3], dtype=complex)

            if iter > 1:
                # Array = np.append(Array, Omega_Max) # Omega_Max will be in Array already, since it is chosen from Array in the first place.
                Array = np.unique(np.append(Array, Omega_Max))
                # Getting indices where Array == Omega_Max:

                mask = np.asarray([1 if i not in PODArray else 0 for i in Array])

                NumberofFrequencies = len(Array)
                TensorArray = np.zeros([NumberofFrequencies, 9], dtype=complex)
                EigenValues = np.zeros([NumberofFrequencies, 3], dtype=complex)

            np.save('Results/' + sweepname + f'/Data/FrequencyArray_iter{iter}.npy', Array)

        print(' performing SVD              ', end='\r')
        cutoff, u1Truncated, u2Truncated, u3Truncated = Truncated_SVD(NumberofSnapshots, PODTol, Theta1Sols)
        plt.savefig('Results/' + sweepname + f'/Graphs/SVD_Decay_iter{iter}.pdf')

        print(f' Number of retained modes = {u1Truncated.shape[1]}')

        save_U = False
        if save_U is True:
            np.save('Results/' + sweepname + '/Data/U1_truncated', u1Truncated)
            np.save('Results/' + sweepname + '/Data/U2_truncated', u2Truncated)
            np.save('Results/' + sweepname + '/Data/U3_truncated', u3Truncated)
            np.save('Results/' + sweepname + '/Data/Theta0', Theta0Sol)
            np.savetxt('Results/' + sweepname + '/Data/PODTensors.csv', PODTensors, delimiter=',')
            np.savetxt('Results/' + sweepname + '/Data/PODEigenvalues.csv', PODEigenValues, delimiter=',')


        ########################################################################
        # Create the ROM

        a0, a1, r1, r2, r3, read_vec, u, v, write_vec = Construct_ROM(Additional_Int_Order, BigProblem, Mu0, Theta0Sol,
                                                                      alpha, epsi, fes, fes2, inout, mu_inv, sigma,
                                                                      xivec)

        if PODErrorBars is True:
            HA0H1, HA0H2, HA0H3, HA1H1, HA1H2, HA1H3, HR1, HR2, HR3, ProL, RerrorReduced1, RerrorReduced2, RerrorReduced3, fes0, ndof0 = Construct_Linear_System(
                PODErrorBars, a0, a1, cutoff, dom_nrs_metal, fes2, mesh, ndof2, r1, r2, r3, read_vec, u1Truncated,
                u2Truncated,
                u3Truncated, write_vec)
        else:
            HA0H1, HA0H2, HA0H3, HA1H1, HA1H2, HA1H3, HR1, HR2, HR3, _, _, _, _, _, _ = Construct_Linear_System(
                PODErrorBars, a0, a1, cutoff, dom_nrs_metal, fes2, mesh, ndof2, r1, r2, r3, read_vec, u1Truncated,
                u2Truncated,
                u3Truncated, write_vec)

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

            G1 = np.transpose(np.conjugate(RerrorReduced1)) @ MR1
            G2 = np.transpose(np.conjugate(RerrorReduced2)) @ MR2
            G3 = np.transpose(np.conjugate(RerrorReduced3)) @ MR3
            G12 = np.transpose(np.conjugate(RerrorReduced1)) @ MR2
            G13 = np.transpose(np.conjugate(RerrorReduced1)) @ MR3
            G23 = np.transpose(np.conjugate(RerrorReduced2)) @ MR3

            G_Store = np.zeros([2 * cutoff + 1, 2 * cutoff + 1, 6], dtype=complex)
            G_Store[:, :, 0] = G1
            G_Store[:, :, 1] = G2
            G_Store[:, :, 2] = G3
            G_Store[:, :, 3] = G12
            G_Store[:, :, 4] = G13
            G_Store[:, :, 5] = G23


        ########################################################################
        # Produce the sweep using the lower dimensional space
        # Setup variables for calculating tensors
        Theta_0j = GridFunction(fes)
        Theta_1i = GridFunction(fes2)
        Theta_1j = GridFunction(fes2)

        if PODErrorBars == True:
            rom1 = np.zeros([2 * cutoff + 1, 1], dtype=complex)
            rom2 = np.zeros([2 * cutoff + 1, 1], dtype=complex)
            rom3 = np.zeros([2 * cutoff + 1, 1], dtype=complex)
            ErrorTensors = np.zeros([NumberofFrequencies, 6])

        # Cleaning up Theta0Sols and Theta1Sols
        # del Theta1Sols
        # del Theta0Sol


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
                Runlist.append(
                    (Core_Distribution[i], mesh, fes, fes2, Lower_Sols[i], u1Truncated, u2Truncated, u3Truncated,
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
                                UAt0U_array, T_array, EU_array_conj, UH_array, Lower_Sols[i], G_Store, cutoff,
                                fes2.ndof,
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
                Distributed_Errors = calc_error_certificates(Core_Distribution[i], alphaLB, G_Store, cutoff, alpha,
                                                             Lower_Sols[i])
                ErrorTensors[Count_Distribution[i], :] = Distributed_Errors

        ### Plotting Updated POD Tensor Coefficients:
        if PlotUpdatedPOD is True:
            cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan']
            max_index_plotting = np.where(Array == x_max)[0][0] + 1 # Adding 1 here so that array contains endpoint
            min_index_plotting = np.where(Array == x_min)[0][0]

            # Plotting Real component:
            plt.figure()
            count = 0
            for i in range(3):
                for j in range(i+1):
                    d = TensorArray.real.reshape(len(Array), 3, 3)
                    plt.semilogx(Array[min_index_plotting:max_index_plotting],
                                 d[min_index_plotting:max_index_plotting, i, j],
                                 label=f'MM: i={i + 1},j={j + 1}', color=cols[count])
                    count += 1
            count = 0
            for i in range(3):
                for j in range(i+1):
                    d_pod = PODTensors.real.reshape(len(PODArray), 3, 3)
                    plt.semilogx(PODArray, d_pod[:, i, j], label=f'SS: i={i + 1}, j={j + 1}', color=cols[count], marker='*', linestyle='None')
                    count += 1

            count = 0
            for i in range(3):
                for j in range(i+1):
                    if i == j == 0:
                        error = ErrorTensors[min_index_plotting:max_index_plotting,0]
                    elif i == j == 1:
                        error = ErrorTensors[min_index_plotting:max_index_plotting,1]
                    elif i == j == 2:
                        error = ErrorTensors[min_index_plotting:max_index_plotting,2]
                    elif i == 0 and j == 1:
                        error = ErrorTensors[min_index_plotting:max_index_plotting,3]
                    elif i == 0 and j == 2:
                        error = ErrorTensors[min_index_plotting:max_index_plotting,4]
                    elif i == 1 and j == 2:
                        error = ErrorTensors[min_index_plotting:max_index_plotting,5]

                    d = TensorArray.real.reshape(len(Array), 3, 3)
                    plt.semilogx(Array[min_index_plotting:max_index_plotting],
                                 np.squeeze(d[min_index_plotting:max_index_plotting, i, j]) + np.squeeze(error),
                                 label=f'Error Certificates: i={i + 1}, j={j + 1}', color=cols[count], linestyle='--')

                    plt.semilogx(Array[min_index_plotting:max_index_plotting],
                                 np.squeeze(d[min_index_plotting:max_index_plotting, i, j]) - np.squeeze(error),
                                 color=cols[count], linestyle='--')
                    count += 1
            plt.legend(title=f'Iteration {iter}', prop={'size': 8}, loc=3)
            plt.xlabel('$\omega$, [rad/s]')
            plt.ylabel(r'$(\tilde{\mathcal{R}})_{ij}$, [m$^3$]')

            # Calculate the limits
            ymin = np.amin(TensorArray.real)
            ymax = np.amax(TensorArray.real)
            y_range = ymax - ymin
            ymin -= 0.05 * y_range
            ymax += 0.05 * y_range
            plt.ylim([ymin, ymax])

            plt.savefig('Results/' + sweepname + f'/Graphs/Real_Tensor_Coeffs_iter{iter}.pdf')


            # Plotting imag component
            plt.figure()
            count = 0
            for i in range(3):
                for j in range(i+1):
                    d = TensorArray.imag.reshape(len(Array), 3, 3)
                    plt.semilogx(Array[min_index_plotting:max_index_plotting],
                                 d[min_index_plotting:max_index_plotting, i, j],
                                 label=f'i={i + 1},j={j + 1}', color=cols[count])
                    count += 1
            count = 0
            for i in range(3):
                for j in range(i+1):
                    d_pod = PODTensors.imag.reshape(len(PODArray), 3, 3)
                    plt.semilogx(PODArray, d_pod[:, i, j], label=f'SS: i={i + 1}, j={j + 1}', color=cols[count], marker='*', linestyle='None')
                    count += 1

            count = 0
            for i in range(3):
                for j in range(i+1):
                    if i == j == 0:
                        error = ErrorTensors[min_index_plotting:max_index_plotting,0]
                    elif i == j == 1:
                        error = ErrorTensors[min_index_plotting:max_index_plotting,1]
                    elif i == j == 2:
                        error = ErrorTensors[min_index_plotting:max_index_plotting,2]
                    elif i == 0 and j == 1:
                        error = ErrorTensors[min_index_plotting:max_index_plotting,3]
                    elif i == 0 and j == 2:
                        error = ErrorTensors[min_index_plotting:max_index_plotting,4]
                    elif i == 1 and j == 2:
                        error = ErrorTensors[min_index_plotting:max_index_plotting,5]

                    d = TensorArray.imag.reshape(len(Array), 3, 3)
                    plt.semilogx(Array[min_index_plotting:max_index_plotting],
                                 np.squeeze(d[min_index_plotting:max_index_plotting, i, j]) + np.squeeze(error),
                                 label=f'Error Certificates: i={i + 1}, j={j + 1}', color=cols[count], linestyle='--')
                    plt.semilogx(Array[min_index_plotting:max_index_plotting],
                                 np.squeeze(d[min_index_plotting:max_index_plotting, i, j]) - np.squeeze(error),
                                 color=cols[count], linestyle='--')
                    count += 1
            plt.legend(title=f'Iteration {iter}', prop={'size': 8}, loc=2)
            plt.xlabel('$\omega$, [rad/s]')
            plt.ylabel(r'$(\mathcal{I})_{ij}$, [m$^3$]')

            # Calculate the limits
            ymin = np.amin(TensorArray.imag)
            ymax = np.amax(TensorArray.imag)
            y_range = ymax - ymin
            ymin -= 0.05 * y_range
            ymax += 0.05 * y_range
            plt.ylim([ymin, ymax])

            plt.savefig('Results/' + sweepname + f'/Graphs/Imag_Tensor_Coeffs_iter{iter}.pdf')


        # Recording original POD Array and solutions:
        if iter == 1:
            PODArray_orig = PODArray
            TensorArray_orig = TensorArray
            EigenValues_orig = EigenValues
            ErrorTensors_orig = ErrorTensors
            PODEigenValues_orig = PODEigenValues
            PODTensors_orig = PODTensors


        # Before we add and new snapshots, we save the results from this iteration.
        np.save('Results/' + sweepname + f'/Data/PODArray_iter{iter}.npy', PODArray)
        np.save('Results/' + sweepname + f'/Data/PODTensors_iter{iter}.npy', PODTensors)
        np.save('Results/' + sweepname + f'/Data/PODEigenValues_iter{iter}.npy', PODEigenValues)
        np.save('Results/' + sweepname + f'/Data/Tensors_iter{iter}.npy', TensorArray)
        np.save('Results/' + sweepname + f'/Data/Errors_iter{iter}.npy', ErrorTensors)


        # Assigining new snapshots. For each interval between adjacent snapshots, calculate max error for the evaluated
        # frequencies and corresponding frequency. If errors are not tending to 0 at the snapshots, do not update list
        # of new snapshots.
        Omega_Max = np.asarray([])
        Max_Interval_Error = np.asarray([])
        for interval in range(len(PODArray)-1):
            # Obtaining errors only in interval between two snapshots.
            lhs_freq = PODArray[interval]
            rhs_freq = PODArray[interval+1]

            # Here I have already included the snapshot frequency array within the evaluation frequency array.
            lhs_index = np.where(Array == lhs_freq)[0][0]
            rhs_index = np.where(Array == rhs_freq)[0][0]

            interval_errors = ErrorTensors[lhs_index:rhs_index, :]
            Max_Interval_Error = np.append(Max_Interval_Error, np.max(interval_errors.ravel()))

            # Here, if errors are not decreasing at snapshot value, ignore interval.
            snapshot_error_tol = Max_Interval_Error[-1]
            if (ErrorTensors[rhs_index, :] < snapshot_error_tol).all() == True:
                Max_Error_index = np.where(ErrorTensors == Max_Interval_Error[-1])[0][0]
                if mask[Max_Error_index] == 1:
                    Omega_Max = np.append(Omega_Max, Array[Max_Error_index])
                else:
                    Max_Interval_Error = np.delete(Max_Interval_Error,-1)  # Removing last element of Max_error so that it is still the same size.
            else:
                print(f"The interval between {Array[lhs_index]:.3e} and {Array[rhs_index]:.3e} rad/s doesn't seem to be decreasing at the snapshots. Skipping for refinement.")
                Max_Interval_Error = np.delete(Max_Interval_Error, -1)  # Removing last element of Max_error so that it is still the same size.

        # If no suitable refinement possible, print warning and take mean of largest 2 snapshots
        if len(Omega_Max) == 0:
            print('No suitable refinement found. Taking mean of largest 2 snapshots to use.')
            Max_Interval_Error = np.asarray([0]) # Padding max error so that arrays are the same length.
            Omega_Max = np.asarray([np.mean(PODArray[-2:])])

        # Sorting frequencies based on corresponding max error.
        indices = np.argsort(Max_Interval_Error)
        Omega_Max = np.asarray([Omega_Max[i] for i in indices])
        Max_Interval_Error = np.asarray([Max_Interval_Error[i] for i in indices])

        Max_Error = [np.max(ErrorTensors.ravel())]

        Omega_Max = Omega_Max[-(N_snaps_per_iter):] # Grabbing frequencies of greatest error

        print(f'Adding Snapshots at omega = {Omega_Max}')

        if np.asarray(Omega_Max).shape == ():
            Omega_Max = np.asarray([Omega_Max])

        if np.max(Max_Error) / Object_Volume < tol:
            break

        # Storing max error and number of snapshots for plotting convergence at the end of the process.
        ind_lower = Array >= x_min
        ind_upper = Array <= x_max
        Error_Array += [np.amax(ErrorTensors[ind_upper * ind_lower])]
        N_Snaps += [len(PODArray)]

        # Computing Additional Snapshot Solution
        PODArray = np.append(PODArray, np.asarray([Omega_Max]))

        # To avoid confusion about indexing and updating arrays, we consturct temporary arrays which are then updated
        # and replace the old arrays completely.
        Theta1Sols_new = np.zeros((ndof2, NumberofSnapshots + len(Omega_Max), 3), dtype=complex)
        Theta1Sols_new[:, 0:NumberofSnapshots, :] = Theta1Sols

        PODTensors_new = np.zeros((NumberofSnapshots + len(Omega_Max), 9), dtype=complex)
        PODTensors_new[0:NumberofSnapshots, :] = PODTensors

        PODEigenValues_new = np.zeros((NumberofSnapshots + len(Omega_Max), 3), dtype=complex)
        PODEigenValues_new[0:NumberofSnapshots, :] = PODEigenValues


        if use_parallel is False:
            for i in range(3):
                for j, o in enumerate(Omega_Max):
                    Theta1Sols_new[:,-(j+1), i] += Theta1(fes,fes2,Theta0Sol[:,i],xivec[i],Order,alpha,nu_no_omega*o,sigma,mu,inout,Tolerance,Maxsteps,epsi,o,i,3,Solver, NumSolverThreads, Additional_Int_Order)
        else:

            # Work out where to send each frequency
            Theta1_CPUs = min(len(Omega_Max), multiprocessing.cpu_count(), CPUs)
            Core_Distribution = []
            Count_Distribution = []
            for i in range(Theta1_CPUs):
                Core_Distribution.append([])
                Count_Distribution.append([])

            # Distribute between the cores
            CoreNumber = 0
            count = 1
            for i, Omega in enumerate(Omega_Max):
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
            for i in range(len(Omega_Max)):
                if PlotPod == True:
                    Runlist.append((np.asarray([Omega_Max[i]]), mesh, fes, fes2, Theta0Sol, xivec, alpha, sigma, mu_inv, inout,
                                    Tolerance, Maxsteps, epsi, Solver, N0, len(Omega_Max), True, True, counter,
                                    BigProblem, Order, NumSolverThreads, Integration_Order, Additional_Int_Order, 'Theta1_Sweep'))
                else:
                    Runlist.append((np.asarray([Omega_Max[i]]), mesh, fes, fes2, Theta0Sol, xivec, alpha, sigma, mu_inv, inout,
                                    Tolerance, Maxsteps, epsi, Solver, N0, len(Omega_Max), True, False, counter,
                                    BigProblem, Order, NumSolverThreads, Integration_Order, Additional_Int_Order, 'Theta1_Sweep'))

            # Run on the multiple cores
            tqdm.tqdm.set_lock(multiprocessing.RLock())
            if ngsglobals.msg_level != 0:
                to = sys.stdout
            else:
                to = os.devnull
            with supress_stdout(to=to):
                with multiprocessing.get_context("spawn").Pool(Theta1_CPUs, initializer=tqdm.tqdm.set_lock,initargs=(tqdm.tqdm.get_lock(),)) as pool:
                    Outputs = list(tqdm.tqdm(pool.imap(imap_version, Runlist), total=len(Runlist), desc='Solving Theta1 Snapshots',
                                  dynamic_ncols=True))

            try:
                pool.terminate()
                print('manually closed pool')
            except:
                print('Pool has already closed.')
            for i in range(len(Outputs)):
                if PlotPod is True:
                    PODEigenValues_new[i+NumberofSnapshots, :] = Outputs[i][1][0]
                    PODTensors_new[i+NumberofSnapshots, :] = Outputs[i][0][0]
                    for j in range(ndof2):
                        Theta1Sols_new[j, i+NumberofSnapshots, :] = Outputs[i][2][j][0]
                else:
                    for j in range(ndof2):
                        Theta1Sols_new[j, i+NumberofSnapshots, :] = Outputs[i][0][j][0]

            print(' solved theta1 problems     ')

        PODTensors = PODTensors_new
        PODEigenValues = PODEigenValues_new
        Theta1Sols = Theta1Sols_new

        # Sorting PODArray, PODEigenValues, and PODTensors
        Indices = np.argsort(PODArray)
        PODArray = np.asarray([PODArray[i] for i in Indices])
        PODTensors = np.asarray([PODTensors[i, :] for i in Indices])
        PODEigenValues = np.asarray([PODEigenValues[i, :] for i in Indices])
        Theta1Sols = Theta1Sols[:, Indices, :]

        print(f' Weighted Error Estimate = {Max_Error[-1] / Object_Volume}, Iteration {iter}')
        # timing_dictionary[f'iter_{iter}_UpdatedTheta1'] = time.time()


    Error_Array += [Max_Error[-1]]
    N_Snaps += [len(PODArray)]

    Final_Evaluation_Array = Array

    # timing_dictionary['Finished'] = time.time()
    # np.save('Results/' + sweepname + '/Data/IterativeTimings.npy', timing_dictionary)

    fig, ax1 = plt.subplots()
    ax1.semilogy(N_Snaps, Error_Array)
    ax2 = ax1.twinx()
    ax2.semilogy(N_Snaps, [E/Object_Volume for E in Error_Array])
    ax1.set_ylabel('$\mathrm{max}(\Delta$)')
    ax2.set_ylabel('$\mathrm{max}(\Delta) / V$')
    ax1.set_xlabel('N Snapshots')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig('Results/' + sweepname + '/Graphs/Convergence.pdf')

    if PlotPod == True:
        return TensorArray, EigenValues, N0, PODTensors, PODEigenValues, numelements, ErrorTensors, (ndof, ndof2), PODArray, PODArray_orig, TensorArray_orig, EigenValues_orig, ErrorTensors_orig, PODEigenValues_orig, PODTensors_orig, N_Snaps, Error_Array, Final_Evaluation_Array, Array_orig
    else:
        return TensorArray, EigenValues, N0, numelements, ErrorTensors, (ndof, ndof2), PODArray, PODArray_orig, TensorArray_orig, EigenValues_orig, ErrorTensors_orig, PODEigenValues_orig, PODTensors_orig, N_Snaps, Error_Array, Final_Evaluation_Array, Array_orig



