import warnings

import numpy as np
from ngsolve import *
import scipy.sparse as sp
import gc
from matplotlib import pyplot as plt
import warnings


# Function definition to solve the Theta0 problem
# Output -The solution to the theta0 problem as a (NGSolve) gridfunction
def Theta0(fes, Order, alpha, mu_inv, inout, e, Tolerance, Maxsteps, epsi, simnumber, Solver, Additional_Int_Order, use_longdouble=True):
    """
    B.A. Wilson, J.Elgy, P.D. Ledger.
    Function to compute theta 0 solution vectors.
    Note: previously this function also computed N0, and there are some leftover arguments.

    Args:
        fes (comp.HCurl): HCurl finite element space for the Theta0 problem.
        Order (int): order of finite element space. Currently not used
        alpha (float): object size scaling. Currently not used
        mu_inv (comp.GridFunction): Grid Function for mu**(-1). Note that for material discontinuities aligning with vertices no interpolation is done
        inout (comp.CoefficientFunction): 1 inside object 0 outside.
        e (list): direction vector
        Tolerance (float): Iterative solver tolerance
        Maxsteps (int): Max iterations for the interative solver
        epsi (float): Small regularisation term
        simnumber (int): i = 1, 2, or 3. Currently not used
        Solver (str): preconditioner. BDDC or local
        Additional_Int_Order (int): additional orders to be considered when assembling linear and bilinear forms. For use with curved elements adn prisms.
        use_longdouble (bool, optional): option to store data using longdouble format. Currently not used. Defaults to True.

    Returns:
        np.ndarray: Theta0 solution vector
    """

    Theta = GridFunction(fes)
    Theta.Set((0, 0, 0), BND)

    # Test and trial functions
    u = fes.TrialFunction()
    v = fes.TestFunction()

    # Create the bilinear form (this is for theta^0 tilda)
    f = LinearForm(fes)
    f += SymbolicLFI(inout * (2 * (1 - mu_inv)) * InnerProduct(e, curl(v)), bonus_intorder=Additional_Int_Order)
    a = BilinearForm(fes, symmetric=True, condense=True)
    a += SymbolicBFI((mu_inv) * (curl(u) * curl(v)), bonus_intorder=Additional_Int_Order)
    a += SymbolicBFI(epsi * (u * v), bonus_intorder=Additional_Int_Order)
    if Solver == "bddc":
        c = Preconditioner(a, "bddc")  # Apply the bddc preconditioner
    a.Assemble()
    f.Assemble()
    if Solver == "local":
        c = Preconditioner(a, "local")  # Apply the local preconditioner
    c.Update()

    # Solve the problem
    f.vec.data += a.harmonic_extension_trans * f.vec
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * Theta.vec
    inverse = CGSolver(a.mat, c.mat, precision=Tolerance, maxsteps=Maxsteps, printrates=True)
    Theta.vec.data += inverse * res
    Theta.vec.data += a.harmonic_extension * Theta.vec
    Theta.vec.data += a.inner_solve * f.vec
    
    Theta_Return = np.zeros([fes.ndof], dtype=np.longdouble)
    Theta_Return[:] = Theta.vec.FV().NumPy()

    # Printing warning if solver didn't converge.
    if inverse.GetSteps() == inverse.maxsteps:
        warnings.warn(f'Solver did not converge within {inverse.maxsteps} iterations. Solution may be inaccurate.')

    del f, a, c, res, u, v, inverse, Theta
    gc.collect()

    return Theta_Return

