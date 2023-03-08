import numpy as np
from ngsolve import *
import scipy.sparse as sp
import gc


# Function definition to solve the Theta1 problem
# Output -The solution to the theta1 problem as a (NGSolve) gridfunction
def Theta1(fes, fes2, Theta0Sol, xi, Order, alpha, nu, sigma, mu, inout, Tolerance, Maxsteps, epsi, Omega, simnumber,
           outof, Solver, num_solver_threads):
    # # print the counter
    # try:  # This is used for the simulations run in parallel
    #     simnumber.value += 1
    #     print(' solving theta1 %d/%d    ' % (floor((simnumber.value) / 3), outof), end='\r', flush=True)
    # except:
    #     try:  # This is for the simulations run consecutively and the single frequency case
    #         print(' solving theta1 %d/%d    ' % (simnumber, outof), end='\r')
    #     except:  # This is for the single frequency case with 3 CPUs
    #         print(' solving the theta1 problem  ', end='\r', flush=True)

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
    f += SymbolicLFI(inout * (-1j) * nu * sigma * InnerProduct(Theta0, v))
    f += SymbolicLFI(inout * (-1j) * nu * sigma * InnerProduct(xi, v))
    a = BilinearForm(fes2, symmetric=True, condense=True)
    a += SymbolicBFI((mu ** (-1)) * InnerProduct(curl(u), curl(v)))
    a += SymbolicBFI((1j) * inout * nu * sigma * InnerProduct(u, v))
    a += SymbolicBFI((1j) * (1 - inout) * epsi * InnerProduct(u, v))
    if Solver == "bddc":
        c = Preconditioner(a, "bddc")  # Apply the bddc preconditioner
    with TaskManager():
        a.Assemble()
        f.Assemble()
    if Solver == "local":
        c = Preconditioner(a, "local")  # Apply the local preconditioner
    c.Update()

    # Solve
    f.vec.data += a.harmonic_extension_trans * f.vec
    res = f.vec.CreateVector()

    with TaskManager():
        res.data = f.vec - a.mat * Theta.vec
        inverse = CGSolver(a.mat, c.mat, precision=Tolerance, maxsteps=Maxsteps)
        Theta.vec.data += inverse * res
        Theta.vec.data += a.inner_solve * f.vec
        Theta.vec.data += a.harmonic_extension * Theta.vec

    Theta_Return = np.zeros([fes2.ndof], dtype=np.clongdouble)
    Theta_Return[:] = Theta.vec.FV().NumPy()

    del f, a, c, res, u, v, Theta
    gc.collect()

    return Theta_Return
