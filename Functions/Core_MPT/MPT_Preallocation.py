import numpy as np
import netgen.meshing as ngmeshing
from ngsolve import *
import sys
from ..PrerunChecks.BilinearForms_Check import *
from ..PrerunChecks.Volume_Consistency_Check import *
sys.path.insert(0,"Settings")
from Settings import PrerunCheckSettings


def MPT_Preallocation(Array, Object, PODArray, curve, inorout, mur, sig, Order, Order_L2, sweepname):
    """
    James Elgy - 2023
    Function to gereate and preallocate arrays, NGSolve meshes and NGSolve coefficient functions based on desired input.
    This function is the same for all modes of MPT calculator, although PODArray can be substituded for an empty list in
    the case of FullSweep, FullSweepMulti, and SingleSolve.

    Parameters
    ----------
    Array: list or numpy array containing the frequencies of interest.
    Object: file path to the object .vol file.
    PODArray: list or numpy array containing the snapshot frequencies for POD.
    curve: int order for curvature of surface elements.
    inorout: dict containing 1 for inside object, 0 for air region. e.g. {'sphere': 1, 'air': 0}
    mur: dictionary containing relative permeability for each region . e.g. {'shell': 0, 'core': 10', 'air' :1}
    sig: dictionary containing conductivity for each region . e.g. {'shell': 1e6, 'core': 6e6', 'air' :0}

    Returns
    -------
    EigenValues,
    Mu0,
    N0,
    NumberofFrequencies,
    NumberofSnapshots,
    TensorArray,
    inout,
    mesh,
    mu,
    numelements,
    sigma

    """


    Object = Object[:-4] + ".vol"

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

    # #if Object[:-4] == 'OCC_sphere_coeff_test':
    # mu = mu * np.load('new_mur_coeff.npy', allow_pickle=True)[0]
    # sigma = sigma * np.load('new_sigma_coeff.npy', allow_pickle=True)[0]


    # if Object[:-4] == 'L2_Check_sphere':
    L2Order = 0
    fesl2 = L2(mesh, order=L2Order)
    sigmaspecial = GridFunction(fesl2)
    muspecial = GridFunction(fesl2)
    # Interpolate grid function on to L2 space before using in bilinear form
    sigmaspecial.Set(sigma)
    muspecial.Set(mu**(-1))
    mu_inv = muspecial
    sigma = sigmaspecial


    # Running pre-sweep checks for integration order and mesh consistency.
    run, bilinear_tol, max_iter = PrerunCheckSettings()
    if run is True:
        bilinear_bonus_int_order = BilinearForms_Check(mesh, Order, mu_inv, sigma, inout, bilinear_tol, max_iter, curve, 2+L2Order, sweepname)
        # check_mesh_volumes(mesh, inout, True, Object, max([2*Order+2, 3*(curve-1)]), curve)

    else:
        # setting default bilinear int order.
        bilinear_bonus_int_order = 3*(curve - 1) - 2*(Order + 1)

    # Set up how the tensor and eigenvalues will be stored
    N0 = np.zeros([3, 3])
    TensorArray = np.zeros([NumberofFrequencies, 9], dtype=complex)
    RealEigenvalues = np.zeros([NumberofFrequencies, 3])
    ImaginaryEigenvalues = np.zeros([NumberofFrequencies, 3])
    EigenValues = np.zeros([NumberofFrequencies, 3], dtype=complex)
    return EigenValues, Mu0, N0, NumberofFrequencies, NumberofSnapshots, TensorArray, inout, mesh, mu_inv, numelements, sigma, bilinear_bonus_int_order

