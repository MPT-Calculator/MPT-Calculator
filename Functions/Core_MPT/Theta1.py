import numpy as np
from ngsolve import *
import scipy.sparse as sp
import gc
import warnings

# Function definition to solve the Theta1 problem
# Output -The solution to the theta1 problem as a (NGSolve) gridfunction
def Theta1(fes, fes2, Theta0Sol, xi, Order, alpha, nu, sigma, mu_inv, inout, Tolerance, Maxsteps, epsi, Omega, simnumber,
           outof, Solver, num_solver_threads, Additional_Int_Order):



    if num_solver_threads != 'default':
        SetNumThreads(num_solver_threads)


    Theta0 = GridFunction(fes)
    Theta0.vec.FV().NumPy()[:] = Theta0Sol
    Theta = GridFunction(fes2)
    Theta.Set((0, 0, 0), BND)

    # Test and trial functions
    u = fes2.TrialFunction()
    v = fes2.TestFunction()

    # Create the bilinear form (this is for the complex conjugate of theta^1)
    f = LinearForm(fes2)
    f += SymbolicLFI(inout * (-1j) * nu * sigma * InnerProduct(Theta0, v), bonus_intorder=Additional_Int_Order)
    f += SymbolicLFI(inout * (-1j) * nu * sigma * InnerProduct(xi, v), bonus_intorder=Additional_Int_Order)
    a = BilinearForm(fes2, symmetric=True, condense=True)
    a += SymbolicBFI((mu_inv) * InnerProduct(curl(u), curl(v)), bonus_intorder=Additional_Int_Order)
    a += SymbolicBFI((1j) * inout * nu * sigma * InnerProduct(u, v), bonus_intorder=Additional_Int_Order)
    a += SymbolicBFI((1j) * (1 - inout) * epsi * InnerProduct(u, v), bonus_intorder=Additional_Int_Order)
    if Solver == "bddc":
        c = Preconditioner(a, "bddc")  # Apply the bddc preconditioner
    with TaskManager():
        a.Assemble()
        f.Assemble()
    if Solver == "local":
        c = Preconditioner(a, "local")  # Apply the local preconditioner
    c.Update()

    # np.save('systemmatrix.npy', [a, f, c])

    # Solve
    f.vec.data += a.harmonic_extension_trans * f.vec
    res = f.vec.CreateVector()

    with TaskManager():
        res.data = f.vec - a.mat * Theta.vec
        inverse = CGSolver(a.mat, c.mat, precision=Tolerance, maxsteps=Maxsteps)
        Theta.vec.data += inverse * res
        Theta.vec.data += a.inner_solve * f.vec
        Theta.vec.data += a.harmonic_extension * Theta.vec

    # Printing warning if solver didn't converge.
    if inverse.GetSteps() == inverse.maxsteps:
        warnings.warn(f'Solver did not converge within {inverse.maxsteps} iterations. Solution may be inaccurate.')



    Theta_Return = np.zeros([fes2.ndof], dtype=np.clongdouble)
    Theta_Return[:] = Theta.vec.FV().NumPy()

    del f, a, c, res, u, v, Theta
    gc.collect()

    return Theta_Return
