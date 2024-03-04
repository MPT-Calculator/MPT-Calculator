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

sys.path.insert(0, "Functions")
from ..Core_MPT.Theta0 import *
from ..Core_MPT.Theta1 import *
from ..Core_MPT.Theta1_Sweep import *
from ..Core_MPT.MPTCalculator import *
from ..Core_MPT.imap_execution import *
from ..Core_MPT.supress_stdout import *

sys.path.insert(0, "Settings")
from Settings import SolverParameters



# Function definition for a full order frequency sweep in parallel
def FullSweepMulti(Object ,Order ,alpha ,inorout ,mur ,sig ,Array ,CPUs ,BigProblem, NumSolverThreads,Integration_Order, Additional_Int_Order, curve=5):
    Object = Object[:-4 ] +".vol"
    # Set up the Solver Parameters
    Solver ,epsi ,Maxsteps ,Tolerance, _, _ = SolverParameters()

    # Loading the object file
    ngmesh = ngmeshing.Mesh(dim=3)
    ngmesh.Load("VolFiles/" +Object)

    # Creating the mesh and defining the element types
    mesh = Mesh("VolFiles/" +Object)
    mesh.Curve(curve  )  # This can be used to refine the mesh
    numelements = mesh.ne  # Count the number elements
    print(" mesh contains  " +str(numelements ) +" elements")

    # Set up the coefficients
    # Scalars
    Mu0 = 4* np.pi * 10 ** (-7)
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
    TensorArray = np.zeros([NumberofFrequencies, 9], dtype=complex)
    RealEigenvalues = np.zeros([NumberofFrequencies, 3])
    ImaginaryEigenvalues = np.zeros([NumberofFrequencies, 3])
    EigenValues = np.zeros([NumberofFrequencies, 3], dtype=complex)

    #########################################################################
    # Theta0
    # This section solves the Theta0 problem to calculate both the inputs for
    # the Theta1 problem and calculate the N0 tensor

    # Setup the finite element space
    # fes = HCurl(mesh, order=Order, dirichlet="outer", flags={"nograds": True})
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes = HCurl(mesh, order=Order, dirichlet="outer", gradientdomains=dom_nrs_metal)

    # Count the number of degrees of freedom
    ndof = fes.ndof

    # Define the vectors for the right hand side
    evec = [CoefficientFunction((1, 0, 0)), CoefficientFunction((0, 1, 0)), CoefficientFunction((0, 0, 1))]

    # Setup the grid functions and array which will be used to save
    Theta0i = GridFunction(fes)
    Theta0j = GridFunction(fes)
    Theta0Sol = np.zeros([ndof, 3])

    # Setup the inputs for the functions to run
    Theta0CPUs = min(3, multiprocessing.cpu_count(), CPUs)
    Runlist = []
    for i in range(3):
        if Theta0CPUs < 3:
            NewInput = (fes, Order, alpha, mu, inout, evec[i], Tolerance, Maxsteps, epsi, i + 1, Solver,Additional_Int_Order, 'Theta0')
        else:
            NewInput = (fes, Order, alpha, mu, inout, evec[i], Tolerance, Maxsteps, epsi, "No Print", Solver, Additional_Int_Order, 'Theta0')
        Runlist.append(NewInput)
    # Run on the multiple cores
    with multiprocessing.get_context("spawn").Pool(Theta0CPUs) as pool:
        Output = list(tqdm.tqdm(pool.map(imap_version, Runlist), total=len(Runlist), desc='Solving Theta0', dynamic_ncols=True))
        # Output = pool.starmap(Theta0, Runlist)
    print(' solved theta0 problems    ')

    # Unpack the outputs
    for i, Direction in enumerate(Output):
        Theta0Sol[:, i] = Direction

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
    # Core_Distribution = []
    # Count_Distribution = []
    # for i in range(Theta1_CPUs):
    #     Core_Distribution.append([])
    #     Count_Distribution.append([])
    #
    # # Distribute between the cores
    # CoreNumber = 0
    # count = 1
    # for i, Omega in enumerate(Array):
    #     Core_Distribution[CoreNumber].append(Omega)
    #     Count_Distribution[CoreNumber].append(i)
    #     if CoreNumber == CPUs - 1 and count == 1:
    #         count = -1
    #     elif CoreNumber == 0 and count == -1:
    #         count = 1
    #     else:
    #         CoreNumber += count

    # Create the inputs
    Runlist = []
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)
    for i in range(len(Array)):
        Runlist.append((np.asarray([Array[i]]), mesh, fes, fes2, Theta0Sol, xivec, alpha, sigma, mu, inout, Tolerance,
                        Maxsteps, epsi, Solver, N0, NumberofFrequencies, False, True, counter, False, Order, NumSolverThreads, Integration_Order, Additional_Int_Order, 'Theta1_Sweep'))


    tqdm.tqdm.set_lock(multiprocessing.RLock())
    # Run on the multiple cores
    if ngsglobals.msg_level != 0:
        to = sys.stdout
    else:
        to = os.devnull
    with supress_stdout(to=to):
        with multiprocessing.get_context("spawn").Pool(Theta1_CPUs, maxtasksperchild=1, initializer=tqdm.tqdm.set_lock, initargs=(tqdm.tqdm.get_lock(),)) as pool:
            Outputs = list(tqdm.tqdm(pool.imap(imap_version, Runlist, chunksize=1), total=len(Runlist), desc='Solving Theta1', dynamic_ncols=True))
        # Outputs = pool.starmap(Theta1_Sweep, Runlist)

    # Unpack the results
    for i in range(len(Outputs)):
        EigenValues[i,:] = Outputs[i][1][0]
        TensorArray[i,:] = Outputs[i][0][0]

    print("Frequency Sweep complete")

    return TensorArray, EigenValues, N0, numelements, (ndof, ndof2)

def init_workers(lock, stream):
    tqdm.tqdm.set_lock(lock)
    sys.stderr = stream