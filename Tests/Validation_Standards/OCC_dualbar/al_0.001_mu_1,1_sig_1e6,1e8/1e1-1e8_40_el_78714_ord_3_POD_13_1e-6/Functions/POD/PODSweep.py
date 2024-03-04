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
sys.path.insert(0,"Settings")
from Settings import SolverParameters, DefaultSettings, IterativePODParameters

# Importing matplotlib for plotting comparisons
import matplotlib
# matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import time


def PODSweep(Object, Order, alpha, inorout, mur, sig, Array, PODArray, PODTol, PlotPod, sweepname, SavePOD,
             PODErrorBars, BigProblem, NumSolverThreads, Integration_Order, Additional_Int_Order, curve=5, recoverymode=False, save_U=False):

    Object = Object[:-4] + ".vol"
    # Set up the Solver Parameters
    Solver, epsi, Maxsteps, Tolerance, _, use_integral = SolverParameters()
    # AdditionalInt *= Order

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
    NumberofSnapshots = len(PODArray)
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
    PODN0Errors = np.zeros([3, 1])
    R = np.zeros([3, 3])
    I = np.zeros([3, 3])
    TensorArray = np.zeros([NumberofFrequencies, 9], dtype=complex)
    EigenValues = np.zeros([NumberofFrequencies, 3], dtype=complex)

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

    # Setup the grid functions and array which will be used to save
    Theta0i = GridFunction(fes)
    Theta0j = GridFunction(fes)
    Theta0Sol = np.zeros([ndof, 3])

    # Here, we either load theta0 or calculate.
    if recoverymode is False:
        # Run in three directions and save in an array for later
        for i in tqdm.tqdm(range(3), desc='Solving Theta0'):
            Theta0Sol[:, i] = Theta0(fes, Order, alpha, mu, inout, evec[i], Tolerance, Maxsteps, epsi, i + 1, Solver, Additional_Int_Order)
        print(' solved theta0 problems   ')
    else:
        try:
            Theta0Sol = np.load('Results/' + sweepname + '/Data/Theta0.npy')
        except FileNotFoundError:
            warn(
                'Could not find theta0 file at:' + ' Results/' + sweepname + '/Data/Theta0.npy \nFalling back to calculation of theta0')
            # Run in three directions and save in an array for later
            for i in range(3):
                Theta0Sol[:, i] = Theta0(fes, Order, alpha, mu, inout, evec[i], Tolerance, Maxsteps, epsi, i + 1,
                                         Solver, Additional_Int_Order)
            print(' solved theta0 problems   ')
    np.save('Results/' + sweepname + '/Data/Theta0', Theta0Sol)

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

    # Poission Projection to acount for gradient terms:
    u, v = fes.TnT()
    m = BilinearForm(fes)
    m += SymbolicBFI(u * v, bonus_intorder=Additional_Int_Order)
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

    # Copy the tensor
    # N0+=np.transpose(N0-np.eye(3)@N0)

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

        if PlotPod == True:
            PODTensors, PODEigenValues, Theta1Sols[:, :, :] = Theta1_Sweep(PODArray, mesh, fes, fes2, Theta0Sol, xivec,
                                                                           alpha, sigma, mu, inout, Tolerance, Maxsteps,
                                                                           epsi, Solver, N0, NumberofFrequencies, True,
                                                                           True, False, BigProblem, Order, NumSolverThreads, Integration_Order, Additional_Int_Order)
        else:
            Theta1Sols[:, :, :] = Theta1_Sweep(PODArray, mesh, fes, fes2, Theta0Sol, xivec, alpha, sigma, mu, inout,
                                               Tolerance, Maxsteps, epsi, Solver, N0, NumberofFrequencies, True, False,
                                               False, BigProblem, Order, NumSolverThreads, Integration_Order, Additional_Int_Order)
        print(' solved theta1 problems     ')
        #########################################################################
        # POD

        print(' performing SVD              ', end='\r')
        # Perform SVD on the solution vector matrices
        u1Truncated, s1, vh1 = np.linalg.svd(Theta1Sols[:, :, 0], full_matrices=False)
        u2Truncated, s2, vh2 = np.linalg.svd(Theta1Sols[:, :, 1], full_matrices=False)
        u3Truncated, s3, vh3 = np.linalg.svd(Theta1Sols[:, :, 2], full_matrices=False)
        # Print an update on progress
        print(' SVD complete      ')

        # scale the value of the modes
        s1norm = s1 / s1[0]
        s2norm = s2 / s2[0]
        s3norm = s3 / s3[0]

        # Decide where to truncate
        cutoff = NumberofSnapshots
        for i in range(NumberofSnapshots):
            if s1norm[i] < PODTol:
                if s2norm[i] < PODTol:
                    if s3norm[i] < PODTol:
                        cutoff = i
                        break

        # Truncate the SVD matrices
        u1Truncated = u1Truncated[:, :cutoff]
        u2Truncated = u2Truncated[:, :cutoff]
        u3Truncated = u3Truncated[:, :cutoff]

        print(f' Number of retained modes = {u1Truncated.shape[1]}')

        plt.figure()
        plt.semilogy(s1norm, label='i=1')
        plt.semilogy(s2norm, label='i=2')
        plt.semilogy(s3norm, label='i=3')
        plt.xlabel('Mode')
        plt.ylabel('Normalised Singular Values')
        plt.legend()

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

    print(' creating reduced order model', end='\r')
    nu_no_omega = Mu0 * (alpha ** 2)

    Theta_0 = GridFunction(fes)
    u, v = fes2.TnT()

    if BigProblem == True:
        a0 = BilinearForm(fes2, symmetric=True)
    else:
        a0 = BilinearForm(fes2, symmetric=True)
    a0 += SymbolicBFI((mu ** (-1)) * InnerProduct(curl(u), curl(v)), bonus_intorder=Additional_Int_Order)
    a0 += SymbolicBFI((1j) * (1 - inout) * epsi * InnerProduct(u, v), bonus_intorder=Additional_Int_Order)
    if BigProblem == True:
        a1 = BilinearForm(fes2, symmetric=True)
    else:
        a1 = BilinearForm(fes2, symmetric=True)
    a1 += SymbolicBFI((1j) * inout * nu_no_omega * sigma * InnerProduct(u, v), bonus_intorder=Additional_Int_Order)

    a0.Assemble()
    a1.Assemble()

    Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, 0]
    r1 = LinearForm(fes2)
    r1 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0, v), bonus_intorder=Additional_Int_Order)
    r1 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[0], v), bonus_intorder=Additional_Int_Order)
    r1.Assemble()
    read_vec = r1.vec.CreateVector()
    write_vec = r1.vec.CreateVector()

    Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, 1]
    r2 = LinearForm(fes2)
    r2 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0, v), bonus_intorder=Additional_Int_Order)
    r2 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[1], v), bonus_intorder=Additional_Int_Order)
    r2.Assemble()

    Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, 2]
    r3 = LinearForm(fes2)
    r3 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0, v), bonus_intorder=Additional_Int_Order)
    r3 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[2], v), bonus_intorder=Additional_Int_Order)
    r3.Assemble()

    # Preallocation
    if PODErrorBars == True:
        fes0 = HCurl(mesh, order=0, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
        ndof0 = fes0.ndof
        RerrorReduced1 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)
        RerrorReduced2 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)
        RerrorReduced3 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)
        ProH = GridFunction(fes2)
        ProL = GridFunction(fes0)
    ########################################################################
    # Create the ROM
    R1 = r1.vec.FV().NumPy()
    R2 = r2.vec.FV().NumPy()
    R3 = r3.vec.FV().NumPy()
    A0H = np.zeros([ndof2, cutoff], dtype=complex)
    A1H = np.zeros([ndof2, cutoff], dtype=complex)

    # E1
    for i in range(cutoff):
        read_vec.FV().NumPy()[:] = u1Truncated[:, i]
        write_vec.data = a0.mat * read_vec
        A0H[:, i] = write_vec.FV().NumPy()
        write_vec.data = a1.mat * read_vec
        A1H[:, i] = write_vec.FV().NumPy()
    HA0H1 = (np.conjugate(np.transpose(u1Truncated)) @ A0H)
    HA1H1 = (np.conjugate(np.transpose(u1Truncated)) @ A1H)
    HR1 = (np.conjugate(np.transpose(u1Truncated)) @ np.transpose(R1))

    if PODErrorBars == True:
        # This constructs W^(i) = [r, A0 U^(m,i), A1 U^(m,i)]. Below eqn 31 in efficient comp paper.
        ProH.vec.FV().NumPy()[:] = R1
        ProL.Set(ProH)
        RerrorReduced1[:, 0] = ProL.vec.FV().NumPy()[:]
        for i in range(cutoff):
            ProH.vec.FV().NumPy()[:] = A0H[:, i]
            ProL.Set(ProH)
            RerrorReduced1[:, i + 1] = ProL.vec.FV().NumPy()[:]
            ProH.vec.FV().NumPy()[:] = A1H[:, i]
            ProL.Set(ProH)
            RerrorReduced1[:, i + cutoff + 1] = ProL.vec.FV().NumPy()[:]
    # E2
    for i in range(cutoff):
        read_vec.FV().NumPy()[:] = u2Truncated[:, i]
        write_vec.data = a0.mat * read_vec
        A0H[:, i] = write_vec.FV().NumPy()
        write_vec.data = a1.mat * read_vec
        A1H[:, i] = write_vec.FV().NumPy()
    HA0H2 = (np.conjugate(np.transpose(u2Truncated)) @ A0H)
    HA1H2 = (np.conjugate(np.transpose(u2Truncated)) @ A1H)
    HR2 = (np.conjugate(np.transpose(u2Truncated)) @ np.transpose(R2))

    if PODErrorBars == True:
        ProH.vec.FV().NumPy()[:] = R2
        ProL.Set(ProH)
        RerrorReduced2[:, 0] = ProL.vec.FV().NumPy()[:]
        for i in range(cutoff):
            ProH.vec.FV().NumPy()[:] = A0H[:, i]
            ProL.Set(ProH)
            RerrorReduced2[:, i + 1] = ProL.vec.FV().NumPy()[:]
            ProH.vec.FV().NumPy()[:] = A1H[:, i]
            ProL.Set(ProH)
            RerrorReduced2[:, i + cutoff + 1] = ProL.vec.FV().NumPy()[:]
    # E3
    for i in range(cutoff):
        read_vec.FV().NumPy()[:] = u3Truncated[:, i]
        write_vec.data = a0.mat * read_vec
        A0H[:, i] = write_vec.FV().NumPy()
        write_vec.data = a1.mat * read_vec
        A1H[:, i] = write_vec.FV().NumPy()
    HA0H3 = (np.conjugate(np.transpose(u3Truncated)) @ A0H)
    HA1H3 = (np.conjugate(np.transpose(u3Truncated)) @ A1H)
    HR3 = (np.conjugate(np.transpose(u3Truncated)) @ np.transpose(R3))

    if PODErrorBars == True:
        ProH.vec.FV().NumPy()[:] = R3
        ProL.Set(ProH)
        RerrorReduced3[:, 0] = ProL.vec.FV().NumPy()[:]
        for i in range(cutoff):
            ProH.vec.FV().NumPy()[:] = A0H[:, i]
            ProL.Set(ProH)
            RerrorReduced3[:, i + 1] = ProL.vec.FV().NumPy()[:]
            ProH.vec.FV().NumPy()[:] = A1H[:, i]
            ProL.Set(ProH)
            RerrorReduced3[:, i + cutoff + 1] = ProL.vec.FV().NumPy()[:]

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

        G1 = np.transpose(np.conjugate(RerrorReduced1)) @ MR1
        G2 = np.transpose(np.conjugate(RerrorReduced2)) @ MR2
        G3 = np.transpose(np.conjugate(RerrorReduced3)) @ MR3
        G12 = np.transpose(np.conjugate(RerrorReduced1)) @ MR2
        G13 = np.transpose(np.conjugate(RerrorReduced1)) @ MR3
        G23 = np.transpose(np.conjugate(RerrorReduced2)) @ MR3

        # Clear the variables
        # RerrorReduced1, RerrorReduced2, RerrorReduced3 = None, None, None
        # MR1, MR2, MR3 = None, None, None
        # fes0, m, c, inverse = None, None, None, None

        fes3 = HCurl(mesh, order=Order, dirichlet="outer", gradientdomains=dom_nrs_metal)
        ndof3 = fes3.ndof
        Omega = Array[0]
        u, v = fes3.TnT()
        amax = BilinearForm(fes3)
        amax += SymbolicBFI((mu ** (-1)) * curl(u) * curl(v), bonus_intorder=Additional_Int_Order)
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
        # print(f'Lower bound alphaLB = {alphaLB} \n')
        # Clear the variables
        fes3, amax, apre, pre, invh1, m = None, None, None, None, None, None
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

    # use_integral = False
    if use_integral is False:
        u, v = fes2.TnT()
        # Constructing 𝐊ᵢⱼ (eqn 7 from paper)
        # For the K bilinear forms, and also later bilinear and linear forms, we specify an integration order specific
        # to the postprocessing. See comment in main.py on the topic.
        u, v = fes2.TnT()
        K = BilinearForm(fes2, symmetric=True)
        K += SymbolicBFI(inout * mu ** (-1) * curl(u) * Conj(curl(v)), bonus_intorder=Integration_Order - 2*(Order+1))
        K += SymbolicBFI((1 - inout) * curl(u) * Conj(curl(v)), bonus_intorder=Integration_Order - 2*(Order+1))
        K.Assemble()
        rows, cols, vals = K.mat.COO()
        del K
        Q = sp.csr_matrix((vals, (rows, cols)))
        del rows, cols, vals
        gc.collect()

        # For faster computation of tensor coefficients, we multiply with Ui before the loop.
        # This computes MxM 𝐊ᴹᵢⱼ. For each of the combinations ij we store the smaller matrix rather than recompute in
        # each case.
        Q11 = np.conj(np.transpose(u1Truncated)) @ Q @ u1Truncated
        Q22 = np.conj(np.transpose(u2Truncated)) @ Q @ u2Truncated
        Q33 = np.conj(np.transpose(u3Truncated)) @ Q @ u3Truncated
        Q21 = np.conj(np.transpose(u2Truncated)) @ Q @ u1Truncated
        Q31 = np.conj(np.transpose(u3Truncated)) @ Q @ u1Truncated
        Q32 = np.conj(np.transpose(u3Truncated)) @ Q @ u2Truncated

        del Q
        Q_array = [Q11, Q22, Q33, Q21, Q31, Q32]


        # Similar for 𝐂ᴹᵢⱼ. refered to as A in code. For each of the combinations ij we store the smaller matrix rather
        # than recompute in each case.
        # Using the same basis functions for both the theta0 and theta1 problems allows us to reduce the number of
        # bilinear forms that need to be constructed.
        # For 𝐍ᴷ = (𝐍₀)ᴷ then 𝐂 = 𝐂¹ = 𝐂² and 𝐬ᵢ = 𝐭ᵢ. In this way we only need to consider 𝐂 (called A in code), 𝐬
        # (called E in code) and c (called G in code) from paper.
        A = BilinearForm(fes2, symmetric=True)
        A += SymbolicBFI(sigma * inout * (v * u), bonus_intorder=Integration_Order - 2*(Order+1))
        A.Assemble()
        rows, cols, vals = A.mat.COO()
        del A
        A_mat = sp.csr_matrix((vals, (rows, cols)))

        del rows, cols, vals
        gc.collect()

        E = np.zeros((3, fes2.ndof), dtype=complex)
        G = np.zeros((3, 3))

        for i in range(3):

            E_lf = LinearForm(fes2)
            E_lf += SymbolicLFI(sigma * inout * xivec[i] * v, bonus_intorder=Integration_Order - 2*(Order+1))
            E_lf.Assemble()
            E[i, :] = E_lf.vec.FV().NumPy()[:]
            del E_lf

            for j in range(3):
                G[i, j] = Integrate(sigma * inout * xivec[i] * xivec[j], mesh, order=Integration_Order)

        H = E.transpose()

        print(' Built K, Q, E, and G')

        #Testing:
        # run_test_comparison(u,v, sigma, xivec, inout, mesh, Theta0Sol, Lower_Sols, u1Truncated, fes, fes2)

        # Similarly for the imaginary part, we multiply with the theta0 sols beforehand.
        A_mat_t0_1 = (A_mat) @ Theta0Sol[:, 0]
        A_mat_t0_2 = (A_mat) @ Theta0Sol[:, 1]
        A_mat_t0_3 = (A_mat) @ Theta0Sol[:, 2]


        # (𝐂)^M being the reduced MxM complex matrix. Similarly to the real part, we store each combination of i,j.
        T11 = np.conj(np.transpose(u1Truncated)) @ A_mat @ u1Truncated
        T22 = np.conj(np.transpose(u2Truncated)) @ A_mat @ u2Truncated
        T33 = np.conj(np.transpose(u3Truncated)) @ A_mat @ u3Truncated
        T21 = np.conj(np.transpose(u2Truncated)) @ A_mat @ u1Truncated
        T31 = np.conj(np.transpose(u3Truncated)) @ A_mat @ u1Truncated
        T32 = np.conj(np.transpose(u3Truncated)) @ A_mat @ u2Truncated

        T_array = [T11, T22, T33, T21, T31, T32]

        # At this point, we have constructed each of the main matrices we need and obtained the reduced A matrix. The
        # larger bilinear form can therefore be removed to save memory.
        del A_mat

        At0_array = [A_mat_t0_1, A_mat_t0_2, A_mat_t0_3]

        # Here we compute (𝐨ⱼ)ᵀ (̅𝐂²)ᴹ
        # Renamed to better fit naming convention
        UAt011_conj = np.conj(u1Truncated.transpose()) @ A_mat_t0_1
        UAt022_conj = np.conj(u2Truncated.transpose()) @ A_mat_t0_2
        UAt033_conj = np.conj(u3Truncated.transpose()) @ A_mat_t0_3
        UAt012_conj = np.conj(u1Truncated.transpose()) @ A_mat_t0_2
        UAt013_conj = np.conj(u1Truncated.transpose()) @ A_mat_t0_3
        UAt023_conj = np.conj(u2Truncated.transpose()) @ A_mat_t0_3

        UAt0_conj = [UAt011_conj, UAt022_conj, UAt033_conj, UAt012_conj, UAt013_conj, UAt023_conj]


        # Similarly we compute and store (𝐨ⱼ)ᵀ (𝐂²)ᴹ
        UAt011 = (u1Truncated.transpose()) @ A_mat_t0_1
        UAt022 = (u2Truncated.transpose()) @ A_mat_t0_2
        UAt033 = (u3Truncated.transpose()) @ A_mat_t0_3
        UAt021 = (u2Truncated.transpose()) @ A_mat_t0_1
        UAt031 = (u3Truncated.transpose()) @ A_mat_t0_1
        UAt032 = (u3Truncated.transpose()) @ A_mat_t0_2
        UAt0U_array = [UAt011, UAt022, UAt033, UAt021, UAt031, UAt032]


        # Finally, we can construct constants that do not depend on frequency.
        # the constant c1 corresponds to 𝐨ⱼᵀ 𝐂⁽¹⁾ 𝐨ᵢ. Similar to other cases we store each combination of i and j.
        c1_11 = (np.transpose(Theta0Sol[:, 0])) @ A_mat_t0_1
        c1_22 = (np.transpose(Theta0Sol[:, 1])) @ A_mat_t0_2
        c1_33 = (np.transpose(Theta0Sol[:, 2])) @ A_mat_t0_3
        c1_21 = (np.transpose(Theta0Sol[:, 1])) @ A_mat_t0_1
        c1_31 = (np.transpose(Theta0Sol[:, 2])) @ A_mat_t0_1
        c1_32 = (np.transpose(Theta0Sol[:, 2])) @ A_mat_t0_2

        # c5 corresponds to 𝐬ᵢᵀ 𝐨ⱼ. Note that E has been transposed here.
        c5_11 = E[0, :] @ Theta0Sol[:, 0]
        c5_22 = E[1, :] @ Theta0Sol[:, 1]
        c5_33 = E[2, :] @ Theta0Sol[:, 2]
        c5_21 = E[1, :] @ Theta0Sol[:, 0]
        c5_31 = E[2, :] @ Theta0Sol[:, 0]
        c5_32 = E[2, :] @ Theta0Sol[:, 1]

        # Similarly to other examples we store each combination rather than recompute
        c1_array = [c1_11, c1_22, c1_33, c1_21, c1_31, c1_32]
        c5_array = [c5_11, c5_22, c5_33, c5_21, c5_31, c5_32]

        # c7 = G corresponds to cᵢⱼ from paper. Note that G does not depend on the FEM basis functions, rather is a
        # polynomial.
        c7 = G

        # c8 corresponds to  𝐬ⱼᵀ 𝐨ᵢ and shold equal c5 for on diagonal entries.
        c8_11 = Theta0Sol[:, 0] @ H[:, 0]
        c8_22 = Theta0Sol[:, 1] @ H[:, 1]
        c8_33 = Theta0Sol[:, 2] @ H[:, 2]
        c8_21 = Theta0Sol[:, 1] @ H[:, 0]
        c8_31 = Theta0Sol[:, 2] @ H[:, 0]
        c8_32 = Theta0Sol[:, 2] @ H[:, 1]

        c8_array = [c8_11, c8_22, c8_33, c8_21, c8_31, c8_32]

        # EU is the reduced linear form for E. Here we compute (̅𝐭ᴹ)ᵀ.
        EU_11 = E[0, :] @ np.conj(u1Truncated)
        EU_22 = E[1, :] @ np.conj(u2Truncated)
        EU_33 = E[2, :] @ np.conj(u3Truncated)
        EU_21 = E[1, :] @ np.conj(u1Truncated)
        EU_31 = E[2, :] @ np.conj(u1Truncated)
        EU_32 = E[2, :] @ np.conj(u2Truncated)

        EU_array_conj = [EU_11, EU_22, EU_33, EU_21, EU_31, EU_32]

        H = E.transpose()

        # also computing  (𝐭ᴹ)ᵀ
        # Renamed to better fit naming convention
        UH_11 = u1Truncated.transpose() @ H[:, 0]
        UH_22 = u2Truncated.transpose() @ H[:, 1]
        UH_33 = u3Truncated.transpose() @ H[:, 2]
        UH_21 = u2Truncated.transpose() @ H[:, 0]
        UH_31 = u3Truncated.transpose() @ H[:, 0]
        UH_32 = u3Truncated.transpose() @ H[:, 1]

        UH_array = [UH_11, UH_22, UH_33, UH_21, UH_31, UH_32]

    for k, omega in enumerate(tqdm.tqdm(Array, desc='Solving Reduced Order Systems')):

        # This part is for obtaining the solutions in the lower dimensional space
        # print(' solving reduced order system %d/%d    ' % (k + 1, NumberofFrequencies), end='\r')
        t1 = time.time()
        g1 = np.linalg.solve(HA0H1 + HA1H1 * omega, HR1 * omega)
        g2 = np.linalg.solve(HA0H2 + HA1H2 * omega, HR2 * omega)
        g3 = np.linalg.solve(HA0H3 + HA1H3 * omega, HR3 * omega)

        # This part projects the problem to the higher dimensional space
        W1 = np.dot(u1Truncated, g1).flatten()
        W2 = np.dot(u2Truncated, g2).flatten()
        W3 = np.dot(u3Truncated, g3).flatten()
        # Calculate the tensors
        nu = omega * Mu0 * (alpha ** 2)
        R = np.zeros([3, 3])
        I = np.zeros([3, 3])

        if use_integral is True:
            for i in range(3):
                Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, i]
                xii = xivec[i]
                if i == 0:
                    Theta_1i.vec.FV().NumPy()[:] = W1
                if i == 1:
                    Theta_1i.vec.FV().NumPy()[:] = W2
                if i == 2:
                    Theta_1i.vec.FV().NumPy()[:] = W3
                for j in range(i + 1):
                    Theta_0j.vec.FV().NumPy()[:] = Theta0Sol[:, j]
                    xij = xivec[j]
                    if j == 0:
                        Theta_1j.vec.FV().NumPy()[:] = W1
                    if j == 1:
                        Theta_1j.vec.FV().NumPy()[:] = W2
                    if j == 2:
                        Theta_1j.vec.FV().NumPy()[:] = W3

                    # Real and Imaginary parts
                    R[i, j] = -(((alpha ** 3) / 4) * Integrate((mu ** (-1)) * (curl(Theta_1j) * Conj(curl(Theta_1i))),
                                                               mesh, order=Integration_Order)).real
                    I[i, j] = ((alpha ** 3) / 4) * Integrate(
                        inout * nu * sigma * ((Theta_1j + Theta_0j + xij) * (Conj(Theta_1i) + Theta_0 + xii)), mesh,
                        order=Integration_Order).real

        else:
            for i in range(3):
                t0i = Theta0Sol[:, i] + 1j * np.zeros(Theta0Sol[:, i].shape)
                if i == 0:
                    gi = g1
                    wi = W1
                elif i == 1:
                    gi = g2
                    wi = W2
                elif i == 2:
                    gi = g3
                    wi = W3

                for j in range(i + 1):
                    t0j = Theta0Sol[:, j] + 1j * np.zeros(Theta0Sol[:, j].shape)
                    if j == 0:
                        gj = g1
                        wj = W1
                    elif j == 1:
                        gj = g2
                        wj = W2
                    elif j == 2:
                        gj = g3
                        wj = W3

                    if i == j:
                        Q = Q_array[i]
                        T = T_array[i]
                        c1 = c1_array[i]
                        c8 = c8_array[i]
                        A_mat_t0 = At0_array[i]
                        At0U = UAt0_conj[i]
                        UAt0 = UAt0U_array[i]
                        c5 = c5_array[i]
                        EU = EU_array_conj[i]
                        EU_notconjed = UH_array[i]
                    elif i == 1 and j == 0:
                        Q = Q_array[3]
                        T = T_array[3]
                        c1 = c1_array[3]
                        At0U = UAt0_conj[3]
                        UAt0 = UAt0U_array[3]
                        c8 = c8_array[3]
                        A_mat_t0 = At0_array[0]
                        c5 = c5_array[3]
                        EU = EU_array_conj[3]
                        EU_notconjed = UH_array[3]
                    elif i == 2 and j == 0:
                        Q = Q_array[4]
                        T = T_array[4]
                        At0U = UAt0_conj[4]
                        UAt0 = UAt0U_array[4]
                        c1 = c1_array[4]
                        c8 = c8_array[4]
                        A_mat_t0 = At0_array[0]
                        c5 = c5_array[4]
                        EU = EU_array_conj[4]
                        EU_notconjed = UH_array[4]
                    elif i == 2 and j == 1:
                        Q = Q_array[5]
                        T = T_array[5]
                        At0U = UAt0_conj[5]
                        UAt0 = UAt0U_array[5]
                        c1 = c1_array[5]
                        c8 = c8_array[5]
                        A_mat_t0 = At0_array[1]
                        c5 = c5_array[5]
                        EU = EU_array_conj[5]
                        EU_notconjed = UH_array[5]
                    # # Looping through non-zero entries in sparse K matrix.
                    # Calc Real Part:
                    A = np.conj(gi[None, :]) @ Q @ (gj)[:, None]
                    R[i, j] = (A * (-alpha ** 3) / 4).real

                    # Calc Imag Part:
                    p1 = np.real(np.conj(gi) @ T @ gj)
                    p2 = np.real(1 * np.conj(gj.transpose()) @ At0U)
                    p2 += np.real(1 * gi.transpose() @ UAt0)
                    p3 = np.real(c8 + c5)
                    p4 = np.real(1 * EU @ np.conj(gj))
                    p4 += np.real(1 * gi @ EU_notconjed)
                    # p4 += np.real(EU.transpose() @ np.conj(gi.transpose()))

                    I[i, j] = np.real((alpha ** 3 / 4) * omega * 4 * np.pi * 1e-7 * alpha ** 2 * (
                                c1 + c7[i, j] + p1 + p2 + p3 + p4))

        R += np.transpose(R - np.diag(np.diag(R))).real
        I += np.transpose(I - np.diag(np.diag(I))).real

        # Save in arrays
        TensorArray[k, :] = (N0 + R + 1j * I).flatten()
        EigenValues[k, :] = np.sort(np.linalg.eigvals(N0 + R)) + 1j * np.sort(np.linalg.eigvals(I))

        if PODErrorBars == True:
            rom1[0, 0] = omega
            rom2[0, 0] = omega
            rom3[0, 0] = omega

            rom1[1:1 + cutoff, 0] = -g1.flatten()
            rom2[1:1 + cutoff, 0] = -g2.flatten()
            rom3[1:1 + cutoff, 0] = -g3.flatten()

            rom1[1 + cutoff:, 0] = -(g1 * omega).flatten()
            rom2[1 + cutoff:, 0] = -(g2 * omega).flatten()
            rom3[1 + cutoff:, 0] = -(g3 * omega).flatten()

            error1 = np.conjugate(np.transpose(rom1)) @ G1 @ rom1
            error2 = np.conjugate(np.transpose(rom2)) @ G2 @ rom2
            error3 = np.conjugate(np.transpose(rom3)) @ G3 @ rom3
            error12 = np.conjugate(np.transpose(rom1)) @ G12 @ rom2
            error13 = np.conjugate(np.transpose(rom1)) @ G13 @ rom3
            error23 = np.conjugate(np.transpose(rom2)) @ G23 @ rom3

            error1 = abs(error1) ** (1 / 2)
            error2 = abs(error2) ** (1 / 2)
            error3 = abs(error3) ** (1 / 2)
            error12 = error12.real
            error13 = error13.real
            error23 = error23.real

            Errors = [error1, error2, error3, error12, error13, error23]

            for j in range(6):
                if j < 3:
                    ErrorTensors[k, j] = ((alpha ** 3) / 4) * (Errors[j] ** 2) / alphaLB
                else:
                    ErrorTensors[k, j] = -2 * Errors[j]
                    if j == 3:
                        ErrorTensors[k, j] += (Errors[0] ** 2) + (Errors[1] ** 2)
                        ErrorTensors[k, j] = ((alpha ** 3) / (8 * alphaLB)) * (
                                    (Errors[0] ** 2) + (Errors[1] ** 2) + ErrorTensors[k, j])
                    if j == 4:
                        ErrorTensors[k, j] += (Errors[0] ** 2) + (Errors[2] ** 2)
                        ErrorTensors[k, j] = ((alpha ** 3) / (8 * alphaLB)) * (
                                    (Errors[0] ** 2) + (Errors[1] ** 2) + ErrorTensors[k, j])
                    if j == 5:
                        ErrorTensors[k, j] += (Errors[1] ** 2) + (Errors[2] ** 2)
                        ErrorTensors[k, j] = ((alpha ** 3) / (8 * alphaLB)) * (
                                    (Errors[0] ** 2) + (Errors[1] ** 2) + ErrorTensors[k, j])

    # print(ErrorTensors)

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


