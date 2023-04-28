import warnings

import numpy as np
from ngsolve import *
import scipy.sparse as sp
import gc
from matplotlib import pyplot as plt
import warnings


# Function definition to solve the Theta0 problem
# Output -The solution to the theta0 problem as a (NGSolve) gridfunction
def Theta0(fes, Order, alpha, mu, inout, e, Tolerance, Maxsteps, epsi, simnumber, Solver, Additional_Int_Order, use_longdouble=True):
    # print the progress
    # try:
    #     print(' solving theta0 %d/3' % (simnumber), end='\r', flush=True)
    # except:
    #     print(' solving the theta0 problem', end='\r', flush=True)

    Theta = GridFunction(fes)
    Theta.Set((0, 0, 0), BND)

    # Test and trial functions
    u = fes.TrialFunction()
    v = fes.TestFunction()

    # Create the bilinear form (this is for theta^0 tilda)
    f = LinearForm(fes)
    f += SymbolicLFI(inout * (2 * (1 - mu ** (-1))) * InnerProduct(e, curl(v)), bonus_intorder=Additional_Int_Order)
    a = BilinearForm(fes, symmetric=True, condense=True)
    a += SymbolicBFI((mu ** (-1)) * (curl(u) * curl(v)), bonus_intorder=Additional_Int_Order)
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
    Theta.vec.data += a.inner_solve * f.vec
    Theta.vec.data += a.harmonic_extension * Theta.vec

    Theta_Return = np.zeros([fes.ndof], dtype=np.longdouble)
    Theta_Return[:] = Theta.vec.FV().NumPy()

    # Printing warning if solver didn't converge.
    if inverse.GetSteps() == inverse.maxsteps:
        warnings.warn(f'Solver did not converge within {inverse.maxsteps} iterations. Solution may be inaccurate.')

    del f, a, c, res, u, v, inverse, Theta
    gc.collect()

    return Theta_Return

