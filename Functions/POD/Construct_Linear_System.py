import numpy as np
from ngsolve import *

def Construct_Linear_System(PODErrorBars, a0, a1, cutoff, dom_nrs_metal, fes2, mesh, ndof2, r1, r2, r3, read_vec,
                            u1Truncated, u2Truncated, u3Truncated, write_vec):
    """
    James Elgy 2023:
    Function to construct the smaller system of linear equations that PODP requires.
    Note that this function was automatically generated from the original code, hence the variable names.
    Parameters
    ----------
    PODErrorBars: bool: controls if error bars are required. If so, we populate RerrorReduced1, RerrorReduced2, and RerrorReduced3
    a0: bilinear form for the left-hand side of the LSE. mu dependent part
    a1: bilinear form for the left-hand side of the LSE. mu independent part
    cutoff: int number of retained modes (M)
    dom_nrs_metal: NGSolve gridfunction for which domains require gradients.
    fes2: Theta1 finite element space.
    mesh: NGSolve mesh
    ndof2: int NDOF in the theta1 problem. Used for preallocation
    r1: Linear form for the right hand side of the LSE for i=1
    r2: Linear form for the right hand side of the LSE for i=2
    r3: Linear form for the right hand side of the LSE for i=3
    read_vec: NGSolve vector of size ndof2. Used for temporarily storing data.
    u1Truncated: complex NdArray of size ndof2xM for left singular matrix for i=1
    u2Truncated: complex NdArray of size ndof2xM for left singular matrix for i=2
    u3Truncated: complex NdArray of size ndof2xM for left singular matrix for i=3
    write_vec: NGSolve vector of size ndof2. Used for temporarily storing data.

    Returns
    -------
    HA0H1, HA0H2, HA0H3, HA1H1, HA1H2, HA1H3. NdArray of size MxM for the reduced left hand side. for i=1, 2, 3
    HR1, HR2, HR3. NdArray of size M for the reduced left hand side.
    ProL, RerrorReduced1, RerrorReduced2, RerrorReduced3, fes0, ndof0: Additional outputs for POD errorbars. if PODErrorBars==0, then outputs are None.
    """


    if PODErrorBars == True:
        fes0 = HCurl(mesh, order=0, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
        ndof0 = fes0.ndof
        RerrorReduced1 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)
        RerrorReduced2 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)
        RerrorReduced3 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)
        ProH = GridFunction(fes2)
        ProL = GridFunction(fes0)
    ########################################################################
    # Create the ROM
    R1 = r1.vec.FV().NumPy()
    R2 = r2.vec.FV().NumPy()
    R3 = r3.vec.FV().NumPy()
    A0H = np.zeros([ndof2, cutoff], dtype=complex)
    A1H = np.zeros([ndof2, cutoff], dtype=complex)
    # E1
    for i in range(cutoff):
        read_vec.FV().NumPy()[:] = u1Truncated[:, i]
        write_vec.data = a0.mat * read_vec
        A0H[:, i] = write_vec.FV().NumPy()
        write_vec.data = a1.mat * read_vec
        A1H[:, i] = write_vec.FV().NumPy()
    HA0H1 = (np.conjugate(np.transpose(u1Truncated)) @ A0H)
    HA1H1 = (np.conjugate(np.transpose(u1Truncated)) @ A1H)
    HR1 = (np.conjugate(np.transpose(u1Truncated)) @ np.transpose(R1))
    if PODErrorBars == True:
        ProH.vec.FV().NumPy()[:] = R1
        ProL.Set(ProH)
        RerrorReduced1[:, 0] = ProL.vec.FV().NumPy()[:]
        for i in range(cutoff):
            ProH.vec.FV().NumPy()[:] = A0H[:, i]
            ProL.Set(ProH)
            RerrorReduced1[:, i + 1] = ProL.vec.FV().NumPy()[:]
            ProH.vec.FV().NumPy()[:] = A1H[:, i]
            ProL.Set(ProH)
            RerrorReduced1[:, i + cutoff + 1] = ProL.vec.FV().NumPy()[:]
    # E2
    for i in range(cutoff):
        read_vec.FV().NumPy()[:] = u2Truncated[:, i]
        write_vec.data = a0.mat * read_vec
        A0H[:, i] = write_vec.FV().NumPy()
        write_vec.data = a1.mat * read_vec
        A1H[:, i] = write_vec.FV().NumPy()
    HA0H2 = (np.conjugate(np.transpose(u2Truncated)) @ A0H)
    HA1H2 = (np.conjugate(np.transpose(u2Truncated)) @ A1H)
    HR2 = (np.conjugate(np.transpose(u2Truncated)) @ np.transpose(R2))
    if PODErrorBars == True:
        ProH.vec.FV().NumPy()[:] = R2
        ProL.Set(ProH)
        RerrorReduced2[:, 0] = ProL.vec.FV().NumPy()[:]
        for i in range(cutoff):
            ProH.vec.FV().NumPy()[:] = A0H[:, i]
            ProL.Set(ProH)
            RerrorReduced2[:, i + 1] = ProL.vec.FV().NumPy()[:]
            ProH.vec.FV().NumPy()[:] = A1H[:, i]
            ProL.Set(ProH)
            RerrorReduced2[:, i + cutoff + 1] = ProL.vec.FV().NumPy()[:]
    # E3
    for i in range(cutoff):
        read_vec.FV().NumPy()[:] = u3Truncated[:, i]
        write_vec.data = a0.mat * read_vec
        A0H[:, i] = write_vec.FV().NumPy()
        write_vec.data = a1.mat * read_vec
        A1H[:, i] = write_vec.FV().NumPy()
    HA0H3 = (np.conjugate(np.transpose(u3Truncated)) @ A0H)
    HA1H3 = (np.conjugate(np.transpose(u3Truncated)) @ A1H)
    HR3 = (np.conjugate(np.transpose(u3Truncated)) @ np.transpose(R3))
    if PODErrorBars == True:
        ProH.vec.FV().NumPy()[:] = R3
        ProL.Set(ProH)
        RerrorReduced3[:, 0] = ProL.vec.FV().NumPy()[:]
        for i in range(cutoff):
            ProH.vec.FV().NumPy()[:] = A0H[:, i]
            ProL.Set(ProH)
            RerrorReduced3[:, i + 1] = ProL.vec.FV().NumPy()[:]
            ProH.vec.FV().NumPy()[:] = A1H[:, i]
            ProL.Set(ProH)
            RerrorReduced3[:, i + cutoff + 1] = ProL.vec.FV().NumPy()[:]

    if PODErrorBars is False:
        RerrorReduced1 = None
        RerrorReduced2 = None
        RerrorReduced3 = None
        fes0 = None
        ndof0 = None
        ProL = None
    return HA0H1, HA0H2, HA0H3, HA1H1, HA1H2, HA1H3, HR1, HR2, HR3, ProL, RerrorReduced1, RerrorReduced2, RerrorReduced3, fes0, ndof0