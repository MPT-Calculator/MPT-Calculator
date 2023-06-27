from ngsolve import *

def Theta0_Postprocessing(Additional_Int_Order, Theta0Sol, fes):
    """
    James Elgy - 2023

    Function to remove gradient terms left over in the theta0 solution, due to using the same finite element space
    as the theta1 problem.

    [1] S. Zaglmayr, “High Order Finite Element Methods for Electromagnetic Field Computation,”
    Johannes Kepler University, 2006.

    Parameters
    ----------
    Additional_Int_Order: int bonus integration order to add to bilinear forms.
    Theta0Sol: NDArray coaining the theta0 solution vectors for each direction.
    fes: Theta0 NGSolve finite element space

    Returns
    -------
    Updated Theta0Sol
    """

    # Poission Projection to acount for gradient terms:
    u, v = fes.TnT()
    m = BilinearForm(fes)
    m += SymbolicBFI(u * v, bonus_intorder=Additional_Int_Order)
    m.Assemble()
    # build gradient matrix as sparse matrix (and corresponding scalar FESpace)
    gradmat, fesh1 = fes.CreateGradient()
    gradmattrans = gradmat.CreateTranspose()  # transpose sparse matrix
    math1 = gradmattrans @ m.mat @ gradmat  # multiply matrices
    math1[0, 0] += 1  # fix the 1-dim kernel
    invh1 = math1.Inverse(inverse="sparsecholesky")
    # build the Poisson projector with operator Algebra:
    proj = IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ m.mat
    theta0 = GridFunction(fes)
    for i in range(3):
        theta0.vec.FV().NumPy()[:] = Theta0Sol[:, i]
        theta0.vec.data = proj * (theta0.vec)
        Theta0Sol[:, i] = theta0.vec.FV().NumPy()[:]

    return Theta0Sol