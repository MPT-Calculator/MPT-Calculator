import numpy as np
from ngsolve import *
import scipy.sparse as sp
import gc
import warnings

# Function definition to solve the Theta1 problem
# Output -The solution to the theta1 problem as a (NGSolve) gridfunction
def Theta1(fes, fes2, Theta0Sol, xi, Order, alpha, nu, sigma, mu_inv, inout, Tolerance, Maxsteps, epsi, Omega, simnumber,
           outof, Solver, num_solver_threads, Additional_Int_Order):

    
    """
    B.A. Wilson, J.Elgy, P.D. Ledger 2020 - 2024

    Function to compute Theta^(1) for the single frequency case.
    
    1) Preallocation
    2) Assemble frequency independent linear form for i=1,2,3.
    3) Assemble frequency dependent bilinear form.
    4) Assign preconditioner.
    5) Solve for i=1,2,3

    
    Args:
        fes (comp.HCurl): HCurl finite element space for the Theta0 problem.
        fes2 (comp.HCurl): HCurl finite element space for the Theta1 problem.
        Theta0Sol (np.ndarray): ndof x 3 array of theta0 solutions.
        xi (list): ith direction vector
        Order (int): order of finite element space.
        alpha (float): object size scaling
        nu (comp.GridFunction): material parameter nu
        sigma (comp.GridFunction): Grid Function for sigma. Note that for material discontinuities aligning with vertices no interpolation is done
        mu_inv (comp.GridFunction): Grid Function for mu**(-1). Note that for material discontinuities aligning with vertices no interpolation is done
        inout (comp.CoefficientFunction): 1 inside object 0 outside.
        Tolerance (float): Iterative solver tolerance
        Maxsteps (int): Max iterations for the interative solver
        epsi (float): Small regularisation term
        Omega (float): frequency of interest (rad/s)
        simnumber (int): counter to keep track of how many directions have been done
        outof (int): total number of directions (3).
        Solver (str): preconditioner. BDDC or local
        num_solver_threads (int | str): Number of parallel threads to use in iterative solver. If 'default' use all threads.
        Additional_Int_Order (int): additional orders to be considered when assembling linear and bilinear forms. For use with curved elements adn prisms.


    Returns:
        Theta1Sols (np.ndarray): ndofx3 complex solution vectors for i=1,2,3

    """
    

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

    # Solve
    f.vec.data += a.harmonic_extension_trans * f.vec
    res = f.vec.CreateVector()

    with TaskManager():
        res.data = f.vec - a.mat * Theta.vec
        inverse = CGSolver(a.mat, c.mat, precision=Tolerance, maxsteps=Maxsteps)
        Theta.vec.data += inverse * res
        
        Theta.vec.data += a.harmonic_extension * Theta.vec
        Theta.vec.data += a.inner_solve * f.vec


    # Printing warning if solver didn't converge.
    if inverse.GetSteps() == inverse.maxsteps:
        warnings.warn(f'Solver did not converge within {inverse.maxsteps} iterations. Solution may be inaccurate.')



    Theta_Return = np.zeros([fes2.ndof], dtype=np.clongdouble)
    Theta_Return[:] = Theta.vec.FV().NumPy()

    del f, a, c, res, u, v, Theta
    gc.collect()

    return Theta_Return
