from ngsolve import *

def Calculate_Minf(Integration_Order, Minf, ThetainfSol, Thetainfi, Thetainfj, alpha, mesh, inout):
    """
    P.D. Ledger - 2025

    function to compute Minf for a given thetainf solution.

    Parameters
    ----------
    Integration_Order: int integration order to use for terms containing non fes polynomials.
    N0: 3x3 preallocated N0 array
    ThetainfSol: ndofx3 ndarray for thetainf
    Thetainfi: preallocated NGsolve GridFunction on fes
    Thetainfj: preallocated NGsolve GridFunction on fes
    alpha: float object scaling alpha
    mesh: NGsolve mesh.
    inout: CoefficientFunction 1 inside object 0 outside.

    Returns
    -------
    Minf
    """

    VolConstant = Integrate(inout , mesh, order=Integration_Order)

    for i in range(3):
        Thetainfi.vec.FV().NumPy()[:] = ThetainfSol[:, i]
        for j in range(3):
            Thetainfj.vec.FV().NumPy()[:] = ThetainfSol[:, j]
            if i == j:
                Minf[i, j] = -(alpha ** 3) * (VolConstant + (1 / 4) * (
                    Integrate( (InnerProduct(curl(Thetainfi), curl(Thetainfj))), mesh, order=Integration_Order)))
            else:
                Minf[i, j] = - (alpha ** 3 / 4) * (
                    Integrate(  (InnerProduct(curl(Thetainfi), curl(Thetainfj))), mesh, order=Integration_Order))
    return Minf
