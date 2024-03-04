# This file contains functions used for creating and solving the reduced order model
# Functions -PODP

# Importing
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import multiprocessing_on_dill as multiprocessing
from ngsolve import *
from ..Core_MPT.MPTFunctions import *


# Function definition to use snapshots to produce a full frequency sweep
# Outputs -Solution vectors for a full frequency sweep (3, 1 for each direction) as numpy arrays
def PODP(mesh, fes0, fes, fes2, Theta0SolVec, xivec, alpha, sigma, mu, inout, epsi, Theta1E1Sol, Theta1E2Sol,
         Theta1E3Sol, FrequencyArray, ConstructedFrequencyArray, PODtol, N0Errors, alphaLB, PODErrorBars):
    # Calculate the imaginary tensors in the full order space (boolean)
    ImagTensorFullOrderCalc = True
    # On how many cores do you want to produce the tensors
    CPUs = 4

    # Print an update on progress
    print(' performing SVD', end='\r')
    # Set up some useful constants
    NumberofFrequencies = len(FrequencyArray)
    NumberofConstructedFrequencies = len(ConstructedFrequencyArray)
    ndof = len(Theta1E1Sol)
    ndof2 = fes2.ndof
    ndof0 = fes0.ndof
    Mu0 = 4 * np.pi * 10 ** (-7)

    # Perform SVD on the solution vector matrices
    u1, s1, vh1 = np.linalg.svd(Theta1E1Sol, full_matrices=False)
    u2, s2, vh2 = np.linalg.svd(Theta1E2Sol, full_matrices=False)
    u3, s3, vh3 = np.linalg.svd(Theta1E3Sol, full_matrices=False)
    # Print an update on progress
    print(' SVD complete      ')

    # scale the value of the modes
    s1norm = s1 / s1[0]
    s2norm = s2 / s2[0]
    s3norm = s3 / s3[0]

    # Decide where to truncate
    cutoff = NumberofFrequencies
    for i in range(NumberofFrequencies):
        if s1norm[i] < PODtol:
            if s2norm[i] < PODtol:
                if s3norm[i] < PODtol:
                    cutoff = i
                    break

    print(cutoff)
    print(s1norm)

    # Truncate the SVD matrices
    u1Truncated = u1[:, :cutoff]
    s1Truncated = s1[:cutoff]
    vh1Truncated = vh1[:cutoff, :]

    u2Truncated = u2[:, :cutoff]
    s2Truncated = s2[:cutoff]
    vh2Truncated = vh2[:cutoff, :]

    u3Truncated = u3[:, :cutoff]
    s3Truncated = s3[:cutoff]
    vh3Truncated = vh3[:cutoff, :]

    # Turn s into a matrix
    s1mat = np.diag(s1)
    s1Truncatedmat = np.diag(s1Truncated)
    s2Truncatedmat = np.diag(s2Truncated)
    s3Truncatedmat = np.diag(s3Truncated)

    # Create where the final solution vectors will be saved
    W1 = np.zeros([ndof, NumberofConstructedFrequencies], dtype=complex)
    W2 = np.zeros([ndof, NumberofConstructedFrequencies], dtype=complex)
    W3 = np.zeros([ndof, NumberofConstructedFrequencies], dtype=complex)

    ########################################################################
    # Create the ROM

    # Set up the stiffness matrices and right hand side
    A1Constant = np.zeros([cutoff, cutoff], dtype=complex)
    A1Variable = np.zeros([cutoff, cutoff], dtype=complex)
    R1Variable = np.zeros([cutoff, 1], dtype=complex)

    A2Constant = np.zeros([cutoff, cutoff], dtype=complex)
    A2Variable = np.zeros([cutoff, cutoff], dtype=complex)
    R2Variable = np.zeros([cutoff, 1], dtype=complex)

    A3Constant = np.zeros([cutoff, cutoff], dtype=complex)
    A3Variable = np.zeros([cutoff, cutoff], dtype=complex)
    R3Variable = np.zeros([cutoff, 1], dtype=complex)

    # Print an update on progress
    print(' creating reduced order model', end='\r')
    with TaskManager():
        Mu0 = 4 * np.pi * 10 ** (-7)
        nu = Mu0 * (alpha ** 2)

        Theta0 = GridFunction(fes)

        u = fes0.TrialFunction()
        v = fes0.TestFunction()

        if PODErrorBars == True:
            m = BilinearForm(fes0)
            m += SymbolicBFI(InnerProduct(u, v))
            f = LinearForm(fes0)
            m.Assemble()
            f.Assemble()
            rowsm, colsm, valsm = m.mat.COO()
            M = sp.csc_matrix((valsm, (rowsm, colsm)))

        u = fes2.TrialFunction()
        v = fes2.TestFunction()

        a0 = BilinearForm(fes2)
        a0 += SymbolicBFI((mu ** (-1)) * InnerProduct(curl(u), curl(v)))
        a0 += SymbolicBFI((1j) * (1 - inout) * epsi * InnerProduct(u, v))
        a1 = BilinearForm(fes2)
        a1 += SymbolicBFI((1j) * inout * nu * sigma * InnerProduct(u, v))

        Theta0.vec.FV().NumPy()[:] = Theta0SolVec[:, 0]
        r1 = LinearForm(fes2)
        r1 += SymbolicLFI(inout * (-1j) * nu * sigma * InnerProduct(Theta0, v))
        r1 += SymbolicLFI(inout * (-1j) * nu * sigma * InnerProduct(xivec[0], v))
        r1.Assemble()

        Theta0.vec.FV().NumPy()[:] = Theta0SolVec[:, 1]
        r2 = LinearForm(fes2)
        r2 += SymbolicLFI(inout * (-1j) * nu * sigma * InnerProduct(Theta0, v))
        r2 += SymbolicLFI(inout * (-1j) * nu * sigma * InnerProduct(xivec[1], v))
        r2.Assemble()

        Theta0.vec.FV().NumPy()[:] = Theta0SolVec[:, 2]
        r3 = LinearForm(fes2)
        r3 += SymbolicLFI(inout * (-1j) * nu * sigma * InnerProduct(Theta0, v))
        r3 += SymbolicLFI(inout * (-1j) * nu * sigma * InnerProduct(xivec[2], v))
        r3.Assemble()

        a0.Assemble()
        a1.Assemble()

        rows0, cols0, vals0 = a0.mat.COO()
        rows1, cols1, vals1 = a1.mat.COO()

    A0 = sp.csr_matrix((vals0, (rows0, cols0)))
    A1 = sp.csr_matrix((vals1, (rows1, cols1)))

    R1 = sp.csr_matrix(r1.vec.FV().NumPy())
    R2 = sp.csr_matrix(r2.vec.FV().NumPy())
    R3 = sp.csr_matrix(r3.vec.FV().NumPy())

    H1 = sp.csr_matrix(u1Truncated)
    H2 = sp.csr_matrix(u2Truncated)
    H3 = sp.csr_matrix(u3Truncated)

    A0H1 = A0 @ H1
    A1H1 = A1 @ H1
    A0H2 = A0 @ H2
    A1H2 = A1 @ H2
    A0H3 = A0 @ H3
    A1H3 = A1 @ H3

    HA0H1 = (np.conjugate(np.transpose(H1)) @ A0H1).todense()
    HA1H1 = (np.conjugate(np.transpose(H1)) @ A1H1).todense()
    HR1 = (np.conjugate(np.transpose(H1)) @ np.transpose(R1)).todense()

    HA0H2 = (np.conjugate(np.transpose(H2)) @ A0H2).todense()
    HA1H2 = (np.conjugate(np.transpose(H2)) @ A1H2).todense()
    HR2 = (np.conjugate(np.transpose(H2)) @ np.transpose(R2)).todense()

    HA0H3 = (np.conjugate(np.transpose(H3)) @ A0H3).todense()
    HA1H3 = (np.conjugate(np.transpose(H3)) @ A1H3).todense()
    HR3 = (np.conjugate(np.transpose(H3)) @ np.transpose(R3)).todense()

    print(' created reduced order model    ')
    if PODErrorBars == True:
        print(' calculating error bars for reduced order model')

        Rerror1 = np.zeros([ndof2, cutoff * 2 + 1], dtype=complex)
        Rerror2 = np.zeros([ndof2, cutoff * 2 + 1], dtype=complex)
        Rerror3 = np.zeros([ndof2, cutoff * 2 + 1], dtype=complex)

        RerrorReduced1 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)
        RerrorReduced2 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)
        RerrorReduced3 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)

        Rerror1[:, 0] = R1.todense()
        Rerror2[:, 0] = R2.todense()
        Rerror3[:, 0] = R3.todense()

        Rerror1[:, 1:cutoff + 1] = A0H1.todense()
        Rerror2[:, 1:cutoff + 1] = A0H2.todense()
        Rerror3[:, 1:cutoff + 1] = A0H3.todense()

        Rerror1[:, cutoff + 1:] = A1H1.todense()
        Rerror2[:, cutoff + 1:] = A1H2.todense()
        Rerror3[:, cutoff + 1:] = A1H3.todense()

        MR1 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)
        MR2 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)
        MR3 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)

        with TaskManager():
            ProH = GridFunction(fes2)
            ProL = GridFunction(fes0)

            for i in range(2 * cutoff + 1):
                ProH.vec.FV().NumPy()[:] = Rerror1[:, i]
                ProL.Set(ProH)
                RerrorReduced1[:, i] = ProL.vec.FV().NumPy()[:]

                ProH.vec.FV().NumPy()[:] = Rerror2[:, i]
                ProL.Set(ProH)
                RerrorReduced2[:, i] = ProL.vec.FV().NumPy()[:]

                ProH.vec.FV().NumPy()[:] = Rerror3[:, i]
                ProL.Set(ProH)
                RerrorReduced3[:, i] = ProL.vec.FV().NumPy()[:]

        lu = spl.spilu(M, drop_tol=10 ** -4)

        for i in range(2 * cutoff + 1):
            if i == 0:
                MR1 = sp.csr_matrix(lu.solve(RerrorReduced1[:, i]))
                MR2 = sp.csr_matrix(lu.solve(RerrorReduced2[:, i]))
                MR3 = sp.csr_matrix(lu.solve(RerrorReduced3[:, i]))
            else:
                MR1 = sp.vstack((MR1, sp.csr_matrix(lu.solve(RerrorReduced1[:, i]))))
                MR2 = sp.vstack((MR2, sp.csr_matrix(lu.solve(RerrorReduced2[:, i]))))
                MR3 = sp.vstack((MR3, sp.csr_matrix(lu.solve(RerrorReduced3[:, i]))))

        lu = spl.spilu(M, drop_tol=10 ** -4)
        for i in range(2 * cutoff + 1):
            MR1[:, i] = lu.solve(RerrorReduced1[:, i])
            MR2[:, i] = lu.solve(RerrorReduced2[:, i])
            MR3[:, i] = lu.solve(RerrorReduced3[:, i])

        G1 = np.transpose(np.conjugate(RerrorReduced1)) @ np.transpose(MR1)
        G2 = np.transpose(np.conjugate(RerrorReduced2)) @ np.transpose(MR2)
        G3 = np.transpose(np.conjugate(RerrorReduced3)) @ np.transpose(MR3)
        G12 = np.transpose(np.conjugate(RerrorReduced1)) @ np.transpose(MR2)
        G13 = np.transpose(np.conjugate(RerrorReduced1)) @ np.transpose(MR3)
        G23 = np.transpose(np.conjugate(RerrorReduced2)) @ np.transpose(MR3)

        rom1 = np.zeros([1 + 2 * cutoff, 1], dtype=complex)
        rom2 = np.zeros([1 + 2 * cutoff, 1], dtype=complex)
        rom3 = np.zeros([1 + 2 * cutoff, 1], dtype=complex)

        TensorErrors = np.zeros([NumberofConstructedFrequencies, 3])
        ErrorTensors = np.zeros([NumberofConstructedFrequencies, 6])
        ErrorTensor = np.zeros([3, 3])

    ########################################################################
    # project the calculations for tensors to the reduced basis

    with TaskManager():
        # Check whether these are needed
        u = fes2.TrialFunction()
        v = fes2.TestFunction()

        # Real Tensor
        k = BilinearForm(fes2)
        k += SymbolicBFI((mu ** (-1)) * InnerProduct(curl(u), curl(v)))
        k.Assemble()
        rowsk, colsk, valsk = k.mat.COO()
        K = sp.csr_matrix((valsk, (rowsk, colsk)))

        # Imaginary Tensor
        # t4
        T4 = BilinearForm(fes2)
        T4 += SymbolicBFI(inout * sigma * InnerProduct(u, v))
        T4.Assemble()
        rowst, colst, valst = T4.mat.COO()
        T4 = sp.csr_matrix((valst, (rowst, colst)))

        Theta0i = GridFunction(fes)
        Theta0j = GridFunction(fes)

        # t1,2,3
        # 11

        Theta0i.vec.FV().NumPy()[:] = Theta0SolVec[:, 0]
        Theta0j.vec.FV().NumPy()[:] = Theta0SolVec[:, 0]
        ta11 = Integrate(sigma * inout * InnerProduct(Theta0j + xivec[0], Theta0i + xivec[0]), mesh)  # t_1^11

        Tb1 = LinearForm(fes2)  # T_2^1j
        Tb1 += SymbolicLFI(sigma * inout * InnerProduct(v, Theta0i + xivec[0]))
        Tb1.Assemble()

        Tc1 = LinearForm(fes2)  # T_3^1i
        Tc1 += SymbolicLFI(sigma * inout * InnerProduct(Conj(v), Theta0j + xivec[0]))
        Tc1.Assemble()

        # 12

        Theta0j.vec.FV().NumPy()[:] = Theta0SolVec[:, 1]
        ta12 = Integrate(sigma * inout * InnerProduct(Theta0j + xivec[1], Theta0i + xivec[0]), mesh)  # t_1^12

        Tc2 = LinearForm(fes2)  # T_3^2i
        Tc2 += SymbolicLFI(sigma * inout * InnerProduct(Conj(v), Theta0j + xivec[1]))
        Tc2.Assemble()

        # 13

        Theta0j.vec.FV().NumPy()[:] = Theta0SolVec[:, 2]
        ta13 = Integrate(sigma * inout * InnerProduct(Theta0j + xivec[2], Theta0i + xivec[0]), mesh)  # t_1^13

        Tc3 = LinearForm(fes2)  # T_3^3i
        Tc3 += SymbolicLFI(sigma * inout * InnerProduct(Conj(v), Theta0j + xivec[2]))
        Tc3.Assemble()

        # 22

        Theta0i.vec.FV().NumPy()[:] = Theta0SolVec[:, 1]
        Theta0j.vec.FV().NumPy()[:] = Theta0SolVec[:, 1]
        ta22 = Integrate(sigma * inout * InnerProduct(Theta0j + xivec[1], Theta0i + xivec[1]), mesh)  # t_1^22

        Tb2 = LinearForm(fes2)  # T_2^2j
        Tb2 += SymbolicLFI(sigma * inout * InnerProduct(v, Theta0i + xivec[1]))
        Tb2.Assemble()

        # 23
        Theta0j.vec.FV().NumPy()[:] = Theta0SolVec[:, 2]
        ta23 = Integrate(sigma * inout * InnerProduct(Theta0j + xivec[2], Theta0i + xivec[1]), mesh)  # t_1^23

        # 33

        Theta0i.vec.FV().NumPy()[:] = Theta0SolVec[:, 2]
        Theta0j.vec.FV().NumPy()[:] = Theta0SolVec[:, 2]
        ta33 = Integrate(sigma * inout * InnerProduct(Theta0j + xivec[2], Theta0i + xivec[2]), mesh)  # t_1^33

        Tb3 = LinearForm(fes2)  # T_2^3j
        Tb3 += SymbolicLFI(sigma * inout * InnerProduct(v, Theta0i + xivec[2]))
        Tb3.Assemble()

    Tb1 = sp.csr_matrix(Tb1.vec.FV().NumPy())
    Tb2 = sp.csr_matrix(Tb2.vec.FV().NumPy())
    Tb3 = sp.csr_matrix(Tb3.vec.FV().NumPy())
    Tc1 = sp.csr_matrix(Tc1.vec.FV().NumPy())
    Tc2 = sp.csr_matrix(Tc2.vec.FV().NumPy())
    Tc3 = sp.csr_matrix(Tc3.vec.FV().NumPy())

    K11 = (np.conjugate(np.transpose(H1)) @ K @ H1).todense()
    K12 = (np.conjugate(np.transpose(H1)) @ K @ H2).todense()
    K13 = (np.conjugate(np.transpose(H1)) @ K @ H3).todense()
    K22 = (np.conjugate(np.transpose(H2)) @ K @ H2).todense()
    K23 = (np.conjugate(np.transpose(H2)) @ K @ H3).todense()
    K33 = (np.conjugate(np.transpose(H3)) @ K @ H3).todense()

    T21H1 = (Tb1 @ H1).todense()
    T21H2 = (Tb1 @ H2).todense()
    T21H3 = (Tb1 @ H3).todense()
    T22H2 = (Tb2 @ H2).todense()
    T22H3 = (Tb2 @ H3).todense()
    T23H3 = (Tb3 @ H3).todense()

    H1T31 = (np.conjugate(np.transpose(H1)) @ np.transpose(Tc1)).todense()
    H2T31 = (np.conjugate(np.transpose(H2)) @ np.transpose(Tc1)).todense()
    H3T31 = (np.conjugate(np.transpose(H3)) @ np.transpose(Tc1)).todense()
    H2T32 = (np.conjugate(np.transpose(H2)) @ np.transpose(Tc2)).todense()
    H3T32 = (np.conjugate(np.transpose(H3)) @ np.transpose(Tc2)).todense()
    H3T33 = (np.conjugate(np.transpose(H3)) @ np.transpose(Tc3)).todense()

    T411 = (np.conjugate(np.transpose(H1)) @ T4 @ H1).todense()
    T412 = (np.conjugate(np.transpose(H1)) @ T4 @ H2).todense()
    T413 = (np.conjugate(np.transpose(H1)) @ T4 @ H3).todense()
    T422 = (np.conjugate(np.transpose(H2)) @ T4 @ H2).todense()
    T423 = (np.conjugate(np.transpose(H2)) @ T4 @ H3).todense()
    T433 = (np.conjugate(np.transpose(H3)) @ T4 @ H3).todense()

    RealTensors = np.zeros([len(ConstructedFrequencyArray), 9])
    ImagTensors = np.zeros([len(ConstructedFrequencyArray), 9])

    I1 = np.zeros([3, 3])
    I2 = np.zeros([3, 3])
    I3 = np.zeros([3, 3])
    I4 = np.zeros([3, 3])

    Theta0i = GridFunction(fes)
    Theta0j = GridFunction(fes)
    Theta1i = GridFunction(fes2)
    Theta1j = GridFunction(fes2)

    ########################################################################
    # Produce the sweep on the lower dimensional space
    if ImagTensorFullOrderCalc == True:
        W1 = np.zeros([ndof2, len(ConstructedFrequencyArray)], dtype=complex)
        W2 = np.zeros([ndof2, len(ConstructedFrequencyArray)], dtype=complex)
        W3 = np.zeros([ndof2, len(ConstructedFrequencyArray)], dtype=complex)

    for i, omega in enumerate(ConstructedFrequencyArray):

        # This part is for obtaining the solutions in the lower dimensional space
        print(' solving reduced order system %d/%d    ' % (i, NumberofConstructedFrequencies), end='\r')

        g1 = np.linalg.solve(HA0H1 + HA1H1 * omega, HR1 * omega)
        g2 = np.linalg.solve(HA0H2 + HA1H2 * omega, HR2 * omega)
        g3 = np.linalg.solve(HA0H3 + HA1H3 * omega, HR3 * omega)

        # This part projects the problem to the higher dimensional space
        if ImagTensorFullOrderCalc == True:
            W1[:, i] = np.dot(u1Truncated, g1).flatten()
            W2[:, i] = np.dot(u2Truncated, g2).flatten()
            W3[:, i] = np.dot(u3Truncated, g3).flatten()

        # This part is for obtaining the tensor coefficients in the lower dimensional space

        # Real tensors
        RealTensors[i, 0] = -(np.transpose(np.conjugate(g1)) @ K11 @ g1)[0, 0].real
        RealTensors[i, 1] = -(np.transpose(np.conjugate(g1)) @ K12 @ g2)[0, 0].real
        RealTensors[i, 2] = -(np.transpose(np.conjugate(g1)) @ K13 @ g3)[0, 0].real
        RealTensors[i, 3] = RealTensors[i, 1]
        RealTensors[i, 6] = RealTensors[i, 2]
        RealTensors[i, 4] = -(np.transpose(np.conjugate(g2)) @ K22 @ g2)[0, 0].real
        RealTensors[i, 5] = -(np.transpose(np.conjugate(g2)) @ K23 @ g3)[0, 0].real
        RealTensors[i, 7] = RealTensors[i, 5]
        RealTensors[i, 8] = -(np.transpose(np.conjugate(g3)) @ K33 @ g3)[0, 0].real

        # Imaginary tensors
        I1f = I1.flatten()
        I2f = I2.flatten()
        I3f = I3.flatten()
        I4f = I4.flatten()

        ImagTensors[i, 0] = ta11
        ImagTensors[i, 1] = ta12
        ImagTensors[i, 2] = ta13
        ImagTensors[i, 4] = ta22
        ImagTensors[i, 5] = ta23
        ImagTensors[i, 8] = ta33

        ImagTensors[i, 3] = ImagTensors[i, 1]
        ImagTensors[i, 6] = ImagTensors[i, 2]
        ImagTensors[i, 7] = ImagTensors[i, 5]

        ImagTensors[i, 0] += (T21H1 @ g1)[0, 0].real
        ImagTensors[i, 1] += (T21H2 @ g2)[0, 0].real
        ImagTensors[i, 2] += (T21H3 @ g3)[0, 0].real
        ImagTensors[i, 4] += (T22H2 @ g2)[0, 0].real
        ImagTensors[i, 5] += (T22H3 @ g3)[0, 0].real
        ImagTensors[i, 8] += (T23H3 @ g3)[0, 0].real

        ImagTensors[i, 3] = ImagTensors[i, 1]
        ImagTensors[i, 6] = ImagTensors[i, 2]
        ImagTensors[i, 7] = ImagTensors[i, 5]

        ImagTensors[i, 0] += (np.conjugate(np.transpose(g1)) @ H1T31)[0, 0].real
        ImagTensors[i, 1] += (np.conjugate(np.transpose(g2)) @ H2T31)[0, 0].real
        ImagTensors[i, 2] += (np.conjugate(np.transpose(g3)) @ H3T31)[0, 0].real
        ImagTensors[i, 4] += (np.conjugate(np.transpose(g2)) @ H2T32)[0, 0].real
        ImagTensors[i, 5] += (np.conjugate(np.transpose(g3)) @ H3T32)[0, 0].real
        ImagTensors[i, 8] += (np.conjugate(np.transpose(g3)) @ H3T33)[0, 0].real

        ImagTensors[i, 3] = ImagTensors[i, 1]
        ImagTensors[i, 6] = ImagTensors[i, 2]
        ImagTensors[i, 7] = ImagTensors[i, 5]

        ImagTensors[i, 0] += (np.conjugate(np.transpose(g1)) @ T411 @ g1)[0, 0].real
        ImagTensors[i, 1] += (np.conjugate(np.transpose(g1)) @ T412 @ g2)[0, 0].real
        ImagTensors[i, 2] += (np.conjugate(np.transpose(g1)) @ T413 @ g3)[0, 0].real
        ImagTensors[i, 4] += (np.conjugate(np.transpose(g2)) @ T422 @ g2)[0, 0].real
        ImagTensors[i, 5] += (np.conjugate(np.transpose(g2)) @ T423 @ g3)[0, 0].real
        ImagTensors[i, 8] += (np.conjugate(np.transpose(g3)) @ T433 @ g3)[0, 0].real

        ImagTensors[i, 3] = ImagTensors[i, 1]
        ImagTensors[i, 6] = ImagTensors[i, 2]
        ImagTensors[i, 7] = ImagTensors[i, 5]

        ImagTensors[i, :] = ImagTensors[i, :] * nu * omega

        if PODErrorBars == True:
            rom1[0, 0] = omega
            rom2[0, 0] = omega
            rom3[0, 0] = omega

            rom1[1:1 + cutoff, 0] = -g1.flatten()
            rom2[1:1 + cutoff, 0] = -g2.flatten()
            rom3[1:1 + cutoff, 0] = -g3.flatten()

            rom1[1 + cutoff:, 0] = -(g1 * omega).flatten()
            rom2[1 + cutoff:, 0] = -(g2 * omega).flatten()
            rom3[1 + cutoff:, 0] = -(g3 * omega).flatten()

            error1 = np.conjugate(np.transpose(rom1)) @ G1 @ rom1
            error2 = np.conjugate(np.transpose(rom2)) @ G2 @ rom2
            error3 = np.conjugate(np.transpose(rom3)) @ G3 @ rom3
            error12 = np.conjugate(np.transpose(rom1)) @ G12 @ rom2
            error13 = np.conjugate(np.transpose(rom1)) @ G13 @ rom3
            error23 = np.conjugate(np.transpose(rom2)) @ G23 @ rom3

            error1 = abs(error1) ** (1 / 2)
            error2 = abs(error2) ** (1 / 2)
            error3 = abs(error3) ** (1 / 2)
            error12 = error12.real
            error13 = error13.real
            error23 = error23.real

            Errors = [error1, error2, error3, error12, error13, error23]

            for j in range(6):
                if j < 3:
                    ErrorTensors[i, j] = ((alpha ** 3) / 4) * (Errors[j] ** 2) / alphaLB
                else:
                    ErrorTensors[i, j] = -2 * Errors[j]
                    if j == 3:
                        ErrorTensors[i, j] += (Errors[0] ** 2) + (Errors[1] ** 2)
                        ErrorTensors[i, j] = ((alpha ** 3) / (8 * alphaLB)) * (
                                    (Errors[0] ** 2) + (Errors[1] ** 2) + ErrorTensors[i, j])
                    if j == 4:
                        ErrorTensors[i, j] += (Errors[0] ** 2) + (Errors[2] ** 2)
                        ErrorTensors[i, j] = ((alpha ** 3) / (8 * alphaLB)) * (
                                    (Errors[0] ** 2) + (Errors[1] ** 2) + ErrorTensors[i, j])
                    if j == 5:
                        ErrorTensors[i, j] += (Errors[1] ** 2) + (Errors[2] ** 2)
                        ErrorTensors[i, j] = ((alpha ** 3) / (8 * alphaLB)) * (
                                    (Errors[0] ** 2) + (Errors[1] ** 2) + ErrorTensors[i, j])

    RealTensors = ((alpha ** 3) / 4) * RealTensors
    ImagTensors = ((alpha ** 3) / 4) * ImagTensors

    print(' reduced order systems solved        ')

    # Calculate the imaginary tensors in the full order space if required
    if ImagTensorFullOrderCalc == True:
        # Create the inputs for the calculation of the tensors
        Runlist = []
        manager = multiprocessing.Manager()
        counter = manager.Value('i', 0)
        for i, Omega in enumerate(ConstructedFrequencyArray):
            nu = Omega * Mu0 * (alpha ** 2)
            NewInput = (
            mesh, fes, fes2, W1[:, i], W2[:, i], W3[:, i], Theta0SolVec, xivec, alpha, mu, sigma, inout, nu, counter,
            NumberofConstructedFrequencies)
            Runlist.append(NewInput)

        # Run in parallel
        with multiprocessing.Pool(CPUs) as pool:
            Output = pool.starmap(MPTCalculator, Runlist)
        print(' calculated tensors             ')
        print(' frequency sweep complete')

        # Unpack the outputs
        for i, OutputNumber in enumerate(Output):
            ImagTensors[i, :] = (OutputNumber[1]).flatten()

    if PODErrorBars == True:
        return RealTensors, ImagTensors, ErrorTensors
    else:
        return RealTensors, ImagTensors



