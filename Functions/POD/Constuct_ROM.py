import numpy as np
from ngsolve import *

def Construct_ROM(Additional_Int_Order, BigProblem, Mu0, Theta0Sol, alpha, epsi, fes, fes2, inout, mu_inv, sigma,
                  xivec, NumSolverThreads, drop_tol):
    """
    James Elgy - 2023
    Automatically generated function for the construction of the linear and bilinear forms used in the PODP.

    Parameters
    ----------
    Additional_Int_Order: int additional integration order for linear and bilinear forms
    BigProblem: bool used for big problems where single precision is used.
    Mu0: permeability of freespace = 4pi 1e-7
    Theta0Sol: NdArray for theta0 solution vector
    alpha: flaot object scaling term alpha.
    epsi: float numerical regularisation
    fes: Theta0 finite element space
    fes2: Theta1 finite element space
    inout: Coefficient function (1 inside, 0 outside)
    mu_inv: GridFunction for mu_r **-1
    sigma: GridFunction for conductivity
    xivec: cross product of e_1 and xi.

    Returns
    -------
    a0, a1, r1, r2, r3: bilinear and linear forms for PODP system
    read_vec, u, v, write_vec: NGsolve vectors and trial/test functions. Used later for preallocation.
    """

    print(' creating reduced order model', end='\r')
    if NumSolverThreads != 'default':
        SetNumThreads(NumSolverThreads)    
    # Mu0=4*np.pi*10**(-7)
    nu_no_omega = Mu0 * (alpha ** 2)
    Theta_0 = GridFunction(fes)
    u, v = fes2.TnT()

    if BigProblem == True:
        a0 = BilinearForm(fes2, symmetric=True, bonus_intorder=Additional_Int_Order, delete_zero_elements =drop_tol,keep_internal=False, symmetric_storage=True)
    else:
        a0 = BilinearForm(fes2, symmetric=True, bonus_intorder=Additional_Int_Order, delete_zero_elements =drop_tol,keep_internal=False, symmetric_storage=True)
    a0 += SymbolicBFI((mu_inv) * InnerProduct(curl(u), curl(v)), bonus_intorder=Additional_Int_Order)
    a0 += SymbolicBFI((1j) * (1 - inout) * epsi * InnerProduct(u, v), bonus_intorder=Additional_Int_Order)

    if BigProblem == True:
        a1 = BilinearForm(fes2, symmetric=True, delete_zero_elements =drop_tol,keep_internal=False, symmetric_storage=True)
    else:
        a1 = BilinearForm(fes2, symmetric=True, delete_zero_elements =drop_tol,keep_internal=False, symmetric_storage=True)

    a1 += SymbolicBFI((1j) * inout * nu_no_omega * sigma * InnerProduct(u, v), bonus_intorder=Additional_Int_Order)
    with TaskManager():
        a0.Assemble()
        a1.Assemble()
    Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, 0]
    r1 = LinearForm(fes2)
    r1 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0, v), bonus_intorder=Additional_Int_Order)
    r1 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[0], v), bonus_intorder=Additional_Int_Order)
    with TaskManager():
        r1.Assemble()

    read_vec = r1.vec.CreateVector()
    write_vec = r1.vec.CreateVector()
    Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, 1]
    r2 = LinearForm(fes2)
    r2 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0, v), bonus_intorder=Additional_Int_Order)
    r2 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[1], v), bonus_intorder=Additional_Int_Order)
    with TaskManager():
        r2.Assemble()

    Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, 2]
    r3 = LinearForm(fes2)
    r3 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0, v), bonus_intorder=Additional_Int_Order)
    r3 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[2], v), bonus_intorder=Additional_Int_Order)
    with TaskManager():
        r3.Assemble()
    return a0, a1, r1, r2, r3, read_vec, u, v, write_vec
