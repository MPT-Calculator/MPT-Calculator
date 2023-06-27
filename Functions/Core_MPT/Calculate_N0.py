from ngsolve import *

def Calculate_N0(Integration_Order, N0, Theta0Sol, Theta0i, Theta0j, alpha, mesh, mu_inv):
    """
    James Elgy - 2023

    function to compute N0 for a given theta0 solution.

    Parameters
    ----------
    Integration_Order: int integration order to use for terms containing non fes polynomials.
    N0: 3x3 preallocated N0 array
    Theta0Sol: ndofx3 ndarray for theta0
    Theta0i: preallocated NGsolve GridFunction on fes
    Theta0j: preallocated NGsolve GridFunction on fes
    alpha: float object scaling alpha
    mesh: NGsolve mesh.
    mu: NGsolve CoefficientFunction for relative permeability in each region.

    Returns
    -------
    N0
    """

    VolConstant = Integrate(1 - mu_inv, mesh, order=Integration_Order)
    for i in range(3):
        Theta0i.vec.FV().NumPy()[:] = Theta0Sol[:, i]
        for j in range(3):
            Theta0j.vec.FV().NumPy()[:] = Theta0Sol[:, j]
            if i == j:
                N0[i, j] = (alpha ** 3) * (VolConstant + (1 / 4) * (
                    Integrate(mu_inv * (InnerProduct(curl(Theta0i), curl(Theta0j))), mesh, order=Integration_Order)))
            else:
                N0[i, j] = (alpha ** 3 / 4) * (
                    Integrate(mu_inv * (InnerProduct(curl(Theta0i), curl(Theta0j))), mesh, order=Integration_Order))
    return N0
