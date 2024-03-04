import numpy as np
import netgen.meshing as ngmeshing
from ngsolve import *
import sys
sys.path.insert(0,"Settings")
from Settings import SolverParameters, DefaultSettings, IterativePODParameters

def MPT_Preallocation(Array, Object, PODArray, curve, inorout, mur, sig):
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

    # Set up how the tensor and eigenvalues will be stored
    N0 = np.zeros([3, 3])
    TensorArray = np.zeros([NumberofFrequencies, 9], dtype=complex)
    RealEigenvalues = np.zeros([NumberofFrequencies, 3])
    ImaginaryEigenvalues = np.zeros([NumberofFrequencies, 3])
    EigenValues = np.zeros([NumberofFrequencies, 3], dtype=complex)
    return EigenValues, Mu0, N0, NumberofFrequencies, NumberofSnapshots, TensorArray, inout, mesh, mu, numelements, sigma

