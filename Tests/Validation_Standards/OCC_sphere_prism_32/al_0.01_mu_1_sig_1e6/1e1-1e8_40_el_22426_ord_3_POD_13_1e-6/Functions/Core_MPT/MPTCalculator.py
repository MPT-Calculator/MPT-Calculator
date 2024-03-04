import numpy as np
from ngsolve import *
import scipy.sparse as sp
import gc

# Function definition to calculate MPTs from solution vectors
# Outputs -R as a numpy array
#        -I as a numpy array (This contains real values not imaginary ones)
def MPTCalculator(mesh, fes, fes2, Theta1E1Sol, Theta1E2Sol, Theta1E3Sol, Theta0Sol, xivec, alpha, mu, sigma, inout, nu,
                  tennumber, outof, Order, Integration_Order):
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
            R[i, j] = -(((alpha ** 3) / 4) * Integrate((mu ** (-1)) * (curl(Theta1j) * Conj(curl(Theta1i))), mesh,
                                                       order=Integration_Order)).real
            I[i, j] = ((alpha ** 3) / 4) * Integrate(
                inout * nu * sigma * ((Theta1j + Theta0j + xij) * (Conj(Theta1i) + Theta0i + xii)), mesh,
                order=Integration_Order).real
    R += np.transpose(R - np.diag(np.diag(R))).real
    I += np.transpose(I - np.diag(np.diag(I))).real
    return R, I
