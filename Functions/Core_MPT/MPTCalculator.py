import numpy as np
from ngsolve import *
import scipy.sparse as sp
import gc

# Function definition to calculate MPTs from solution vectors
# Outputs -R as a numpy array
#        -I as a numpy array (This contains real values not imaginary ones)
def MPTCalculator(mesh, fes, fes2, Theta1E1Sol, Theta1E2Sol, Theta1E3Sol, Theta0Sol, xivec, alpha, mu_inv, sigma, inout, nu,
                  tennumber, outof, Order, Integration_Order):
    """
    B.A. Wilson, J.Elgy, P.D. Ledger.
    Function to compute R and I for the case of a single frequency. For use with SingleFrequency.py

    Args:
        mesh (comp.Mesh): ngsolve mesh.
        fes (comp.HCurl): HCurl finite element space for the Theta0 problem.
        fes2 (comp.HCurl): HCurl finite element space for the Theta1 problem.
        Theta1E1Sol (np.ndarray): Theta1 solutions for direction i=1
        Theta1E2Sol (np.ndarray): Theta1 solutions for direction i=2
        Theta1E3Sol (np.ndarray): Theta1 solutions for direction i=3
        Theta0Sols (np.ndarray): ndof x 3 array of theta0 solutions.
        xivec (list): 3x3 list of direction vectors
        alpha (float): object size scaling
        mu_inv (comp.GridFunction): Grid Function for mu**(-1). Note that for material discontinuities aligning with vertices no interpolation is done
        sigma (comp.GridFunction): Grid Function for sigma. Note that for material discontinuities aligning with vertices no interpolation is done
        inout (comp.CoefficientFunction): 1 inside object 0 outside.
        nu (comp.CoefficientFunction): nu parameter for each material.
        tennumber (_type_): _description_
        outof (int): _description_
        Order (int): order of finite element space.
        Integration_Order (int): order of integration to be used when computing tensors.

    Returns:
        R (np.ndarray): 3x3 real part
        I (np.ndarray): 3x3 imag part
    """
    
    
    
    # Print the progress of the sweep
    try:  # This is used for the simulations run in parallel
        tennumber.value += 1
        print(' calculating tensor %d/%d    ' % (tennumber.value, outof), end='\r')
    except:  # This is for the POD run consecutively
        try:
            print(' calculating tensor %d/%d    ' % (tennumber, outof), end='\r')
        except:  # This is for the full sweep run consecutively (no print)
            pass

    R = np.zeros([3, 3])
    I = np.zeros([3, 3])
    Theta0i = GridFunction(fes)
    Theta0j = GridFunction(fes)
    Theta1i = GridFunction(fes2)
    Theta1j = GridFunction(fes2)
    for i in range(3):
        Theta0i.vec.FV().NumPy()[:] = Theta0Sol[:, i]
        xii = xivec[i]
        if i == 0:
            Theta1i.vec.FV().NumPy()[:] = Theta1E1Sol
        if i == 1:
            Theta1i.vec.FV().NumPy()[:] = Theta1E2Sol
        if i == 2:
            Theta1i.vec.FV().NumPy()[:] = Theta1E3Sol
        for j in range(i + 1):
            Theta0j.vec.FV().NumPy()[:] = Theta0Sol[:, j]
            xij = xivec[j]
            if j == 0:
                Theta1j.vec.FV().NumPy()[:] = Theta1E1Sol
            if j == 1:
                Theta1j.vec.FV().NumPy()[:] = Theta1E2Sol
            if j == 2:
                Theta1j.vec.FV().NumPy()[:] = Theta1E3Sol

            # Real and Imaginary parts
            R[i, j] = -(((alpha ** 3) / 4) * Integrate((mu_inv) * (curl(Theta1j) * Conj(curl(Theta1i))), mesh,
                                                       order=Integration_Order)).real
            I[i, j] = ((alpha ** 3) / 4) * Integrate(
                inout * nu * sigma * ((Theta1j + Theta0j + xij) * (Conj(Theta1i) + Theta0i + xii)), mesh,
                order=Integration_Order).real
    R += np.transpose(R - np.diag(np.diag(R))).real
    I += np.transpose(I - np.diag(np.diag(I))).real
    return R, I
