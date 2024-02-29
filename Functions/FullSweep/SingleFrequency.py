# This file contains the function called from the main.py file when Single=True
# Functions -SingleFrequency (Solve for one value of omega)
# Importing
import os
import sys
import time
import multiprocessing as multiprocessing
import tqdm.auto as tqdm
import cmath
import numpy as np

import netgen.meshing as ngmeshing
from ngsolve import *

sys.path.insert(0, "Functions")
from ..Core_MPT.Theta1 import *
from ..Core_MPT.Theta0 import *
from ..Core_MPT.MPTCalculator import *
from ..Core_MPT.imap_execution import *
from ..Saving.FtoS import *
from ..Saving.DictionaryList import *
from ..Core_MPT.MPT_Preallocation import *
from ..Core_MPT.Solve_Theta_0_Problem import *
from ..Core_MPT.Calculate_N0 import *
from ..Core_MPT.Theta0_Postprocessing import *

sys.path.insert(0, "Settings")
from Settings import SolverParameters
import gc

from Functions.Helper_Functions.count_prismatic_elements import count_prismatic_elements


def SingleFrequency(Object, Order, alpha, inorout, mur, sig, Omega, CPUs, VTK, Refine, Integration_Order, Additional_Int_Order, Order_L2, sweepname, drop_tol,
                    curve=5, theta_solutions_only=False, num_solver_threads='default'):

    _, Mu0, _, _, _, _, inout, mesh, mu_inv, numelements, sigma, bilinear_bonus_int_order = MPT_Preallocation([Omega], Object, [], curve, inorout,
                                                                                                              mur, sig, Order, 0, sweepname,
                                                                                                              num_solver_threads, drop_tol)
    # Set up the Solver Parameters
    Solver, epsi, Maxsteps, Tolerance, _, use_integral = SolverParameters()

    # Set up how the tensors will be stored
    N0 = np.zeros([3, 3])
    R = np.zeros([3, 3])
    I = np.zeros([3, 3])

    #########################################################################
    # Theta0
    # This section solves the Theta0 problem to calculate both the inputs for
    # the Theta1 problem and calculate the N0 tensor
    # Here, we have set the solver not to use the recovery mode.
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

    # Setup the finite element space
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes2 = HCurl(mesh, order=Order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    # Count the number of degrees of freedom
    ndof2 = fes2.ndof

    # Define the vectors for the right hand side
    xivec = [CoefficientFunction((0, -z, y)), CoefficientFunction((z, 0, -x)), CoefficientFunction((-y, x, 0))]

    # Setup the array which will be used to store the solution vectors
    Theta1Sol = np.zeros([ndof2, 3], dtype=complex)

    # Set up the inputs for the problem
    Runlist = []
    nu = Omega * Mu0 * (alpha ** 2)
    for i in range(3):
        if CPUs < 3:
            NewInput = (
            fes, fes2, Theta0Sol[:, i], xivec[i], Order, alpha, nu, sigma, mu_inv, inout, Tolerance, Maxsteps, epsi, Omega,
            i + 1, 3, Solver, num_solver_threads, Additional_Int_Order, 'Theta1')
        else:
            NewInput = (
            fes, fes2, Theta0Sol[:, i], xivec[i], Order, alpha, nu, sigma, mu_inv, inout, Tolerance, Maxsteps, epsi, Omega,
            "No Print", 3, Solver, num_solver_threads, Additional_Int_Order, 'Theta1')
        Runlist.append(NewInput)

    # Run on the multiple cores
    with multiprocessing.Pool(CPUs) as pool:
        Output = list(tqdm.tqdm(pool.map(imap_version, Runlist), total=len(Runlist), desc='Solving Theta1'))
    print(' solved theta1 problem       ')

    # Unpack the outputs
    for i, OutputNumber in enumerate(Output):
        Theta1Sol[:, i] = OutputNumber

    if theta_solutions_only == True:
        return Theta0Sol, Theta1Sol

    # Create the VTK output if required
    if VTK == True:
        print(' creating vtk output', end='\r')
        ThetaE1 = GridFunction(fes2)
        ThetaE2 = GridFunction(fes2)
        ThetaE3 = GridFunction(fes2)
        ThetaE1.vec.FV().NumPy()[:] = Output[0]
        ThetaE2.vec.FV().NumPy()[:] = Output[1]
        ThetaE3.vec.FV().NumPy()[:] = Output[2]
        E1Mag = CoefficientFunction(
            sqrt(InnerProduct(ThetaE1.real, ThetaE1.real) + InnerProduct(ThetaE1.imag, ThetaE1.imag)))
        E2Mag = CoefficientFunction(
            sqrt(InnerProduct(ThetaE2.real, ThetaE2.real) + InnerProduct(ThetaE2.imag, ThetaE2.imag)))
        E3Mag = CoefficientFunction(
            sqrt(InnerProduct(ThetaE3.real, ThetaE3.real) + InnerProduct(ThetaE3.imag, ThetaE3.imag)))
        Sols = []
        Sols.append(dom_nrs_metal)
        Sols.append((ThetaE1 * 1j * Omega * sigma).real)
        Sols.append((ThetaE1 * 1j * Omega * sigma).imag)
        Sols.append((ThetaE2 * 1j * Omega * sigma).real)
        Sols.append((ThetaE2 * 1j * Omega * sigma).imag)
        Sols.append((ThetaE3 * 1j * Omega * sigma).real)
        Sols.append((ThetaE3 * 1j * Omega * sigma).imag)
        Sols.append(E1Mag * Omega * sigma)
        Sols.append(E2Mag * Omega * sigma)
        Sols.append(E3Mag * Omega * sigma)

        # Creating Save Name:
        strmur = DictionaryList(mur, False)
        strsig = DictionaryList(sig, True)
        savename = "Results/" + Object[:-4] + f"/al_{alpha}_mu_{strmur}_sig_{strsig}" + "/om_" + FtoS(Omega) + f"_el_{numelements}_ord_{Order}/Data/"
        if Refine == True:
            vtk = VTKOutput(ma=mesh, coefs=Sols,
                            names=["Object", "E1real", "E1imag", "E2real", "E2imag", "E3real", "E3imag", "E1Mag",
                                   "E2Mag", "E3Mag"], filename=savename + Object[:-4], subdivision=3)
        else:
            vtk = VTKOutput(ma=mesh, coefs=Sols,
                            names=["Object", "E1real", "E1imag", "E2real", "E2imag", "E3real", "E3imag", "E1Mag",
                                   "E2Mag", "E3Mag"], filename=savename + Object[:-4], subdivision=0)
        vtk.Do()

        # Compressing vtk output and sending to zip file:
        zipObj = ZipFile(savename + 'VTU.zip', 'w', ZIP_DEFLATED)
        zipObj.write(savename + Object[:-4] + '.vtu', os.path.basename(savename + Object[:-4] + '.vtu'))
        zipObj.close()
        os.remove(savename + Object[:-4] + '.vtu')
        print(' vtk output created     ')

    #########################################################################
    # Calculate the tensor and eigenvalues

    # Create the inputs for the calculation of the tensors
    print(' calculating the tensor  ', end='\r')

    if use_integral is True:

        Runlist = []
        nu = Omega * Mu0 * (alpha ** 2)
        R, I = MPTCalculator(mesh, fes, fes2, Theta1Sol[:, 0], Theta1Sol[:, 1], Theta1Sol[:, 2], Theta0Sol, xivec, alpha,
                             mu_inv, sigma, inout, nu, "No Print", 1, Order, Integration_Order)
        print(' calculated the tensor             ')

        # Unpack the outputs
        MPT = N0 + R + 1j * I
        RealEigenvalues = np.sort(np.linalg.eigvals(N0 + R))
        ImaginaryEigenvalues = np.sort(np.linalg.eigvals(I))
        EigenValues = RealEigenvalues + 1j * ImaginaryEigenvalues

    else:
        Theta1Sols = np.zeros((ndof2, 1, 3), dtype=complex)
        Theta1Sols[:, 0, :] = np.asarray(np.squeeze(Theta1Sol))
        print(' Computing coefficients')

        # I'm aware that pre and post multiplying by identity of size ndof2 is slower than using K and A matrices outright,
        # however this allows us to reuse the Construct_Matrices function rather than add (significantly) more code.
        identity1 = sp.identity(ndof2)
        # Cteate the inputs
        Runlist = []

        for i in range(CPUs):
            Runlist.append((np.asarray([Omega]), mesh, fes, fes2, Theta1Sols, identity1, identity1, identity1,
                            Theta0Sol, xivec, alpha, sigma, mu_inv, inout, N0, 1, [],
                            False, 0, 0, Order, Integration_Order, bilinear_bonus_int_order, use_integral))

        # Run on the multiple cores
        # Edit James Elgy: changed how pool was generated to 'spawn': see
        # https://britishgeologicalsurvey.github.io/science/python-forking-vs-spawn/
        with multiprocessing.get_context('spawn').Pool(CPUs) as pool:
            Outputs = pool.starmap(Theta1_Lower_Sweep, Runlist)

        for i, Output in enumerate(Outputs):
            for j, Num in enumerate([Omega]):
                MPT = Output[0][j]
                EigenValues = Output[1][j]

    # del Theta1i, Theta1j, Theta0i, Theta0j, fes, fes2, Theta0Sol, Theta1Sol
    gc.collect()

    return MPT, EigenValues, N0, numelements, (ndof, ndof2)
