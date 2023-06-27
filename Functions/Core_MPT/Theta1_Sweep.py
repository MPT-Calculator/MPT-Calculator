import numpy as np
import sys
import os

# import the desired library
from ngsolve import *
# unsilence command-line output

import gc
import tqdm



def Theta1_Sweep(Array ,mesh ,fes ,fes2 ,Theta0Sols ,xivec ,alpha ,sigma ,mu_inv ,inout ,Tolerance ,Maxsteps ,epsi ,Solver
                 ,N0 ,TotalNOF ,Vectors ,Tensors ,Multi ,BP, Order, num_solver_threads, Integration_Order, Additional_Int_Order):
    # print(' solving theta1')
    # Setup variables
    Mu0 = 4* np.pi * 10 ** (-7)
    nu_no_omega = Mu0 * (alpha ** 2)
    NOF = len(Array)

    # Setup where to store tensors
    if Tensors == True:
        R = np.zeros([3, 3])
        I = np.zeros([3, 3])
        TensorArray = np.zeros([NOF, 9], dtype=complex)
        EigenValues = np.zeros([NOF, 3], dtype=complex)

    # Setup where to save the solution vectors
    if Vectors == True:
        ndof = fes2.ndof
        if BP == True:
            Theta1Sols = np.zeros([ndof, NOF, 3], dtype=np.complex64)
        else:
            Theta1Sols = np.zeros([ndof, NOF, 3], dtype=complex)

    if num_solver_threads != 'default':
        SetNumThreads(num_solver_threads)

    # To improve performance of the iterative solver, we split the problem over several concurrent threads using the
    # NGSolve TaskManager. Currently we construct the linear forms f1, f2, and f3 and the bilinear form, a, in parallel.
    # We also parallelise the CGSolver and the static condensation. This has lead to significant time saving.

    #TaskManager is called here, to force a consistent shared memory parallelism between the linear forms, grid
    # functions and bilinear form when num_solver_threads != default.
    with TaskManager():
        Theta0i = GridFunction(fes)
        Theta0j = GridFunction(fes)
        Theta1i = GridFunction(fes2)
        Theta1j = GridFunction(fes2)
        Theta1 = GridFunction(fes2)
        Theta2 = GridFunction(fes2)
        Theta3 = GridFunction(fes2)

        # Test and trial functions
        u, v = fes2.TnT()

        # Setup righthand sides
        Theta0i.vec.FV().NumPy()[:] = Theta0Sols[:, 0]
        f1 = LinearForm(fes2)
        f1 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta0i, v), bonus_intorder=Additional_Int_Order)
        f1 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[0], v), bonus_intorder=Additional_Int_Order)
        f1.Assemble()

        Theta0i.vec.FV().NumPy()[:] = Theta0Sols[:, 1]
        f2 = LinearForm(fes2)
        f2 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta0i, v), bonus_intorder=Additional_Int_Order)
        f2 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[1], v), bonus_intorder=Additional_Int_Order)
        f2.Assemble()

        Theta0i.vec.FV().NumPy()[:] = Theta0Sols[:, 2]
        f3 = LinearForm(fes2)
        f3 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta0i, v), bonus_intorder=Additional_Int_Order)
        f3 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[2], v), bonus_intorder=Additional_Int_Order)
        f3.Assemble()

        # Set up a vector for the residual and solving
        res = f1.vec.CreateVector()
        ftemp = f1.vec.CreateVector()

    if Multi is not False:
        enumerator = Array
    else:
        enumerator = tqdm.tqdm(Array, desc='Solving Theta1', total=len(Array))

    for k, Omega in enumerate(enumerator):

        #Create the bilinear form
        a = BilinearForm(fes2, symmetric=True, condense=True)
        a += SymbolicBFI((mu_inv) * InnerProduct(curl(u), curl(v)), bonus_intorder=Additional_Int_Order)
        a += SymbolicBFI((1j) * inout * nu_no_omega * Omega * sigma * InnerProduct(u, v), bonus_intorder=Additional_Int_Order)
        a += SymbolicBFI((1j) * (1 - inout) * epsi * InnerProduct(u, v), bonus_intorder=Additional_Int_Order)

        if Solver == "bddc":
            c = Preconditioner(a, "bddc")  # Apply the bddc preconditioner
        with TaskManager():
            a.Assemble()
        if Solver == "local":
            c = Preconditioner(a, "local")  # Apply the local preconditioner
        # with TaskManager():
        c.Update()

        # Calculate the inverse operator
        with TaskManager():
            inverse = CGSolver(a.mat, c.mat, precision=Tolerance, maxsteps=Maxsteps)

        # Solve in each direction

        Theta1.Set((0, 0, 0), BND)
        Theta2.Set((0, 0, 0), BND)
        Theta3.Set((0, 0, 0), BND)

        # e1
        res.data.FV().NumPy()[:] = f1.vec.FV().NumPy() * Omega
        with TaskManager():
            res.data += a.harmonic_extension_trans * res.data
        ftemp.data = res.data
        with TaskManager():
            res.data -= a.mat * Theta1.vec
            Theta1.vec.data += inverse * res
            Theta1.vec.data += a.inner_solve * ftemp.data
            Theta1.vec.data += a.harmonic_extension * Theta1.vec

        # e2
        res.data.FV().NumPy()[:] = f2.vec.FV().NumPy() * Omega
        with TaskManager():
            res.data += a.harmonic_extension_trans * res.data
        ftemp.data = res.data
        with TaskManager():
            res.data -= a.mat * Theta2.vec
            Theta2.vec.data += inverse * res
            Theta2.vec.data += a.inner_solve * ftemp.data
            Theta2.vec.data += a.harmonic_extension * Theta2.vec

        # e3
        res.data.FV().NumPy()[:] = f3.vec.FV().NumPy() * Omega
        with TaskManager():
            res.data += a.harmonic_extension_trans * res.data
        ftemp.data = res.data
        with TaskManager():
            res.data -= a.mat * Theta3.vec
            Theta3.vec.data += inverse * res
            Theta3.vec.data += a.inner_solve * ftemp.data
            Theta3.vec.data += a.harmonic_extension * Theta3.vec

        if Vectors == True:
            Theta1Sols[:, k, 0] = Theta1.vec.FV().NumPy()
            Theta1Sols[:, k, 1] = Theta2.vec.FV().NumPy()
            Theta1Sols[:, k, 2] = Theta3.vec.FV().NumPy()

        if Tensors == True:
            # Calculate upper triangle of tensor
            R = np.zeros([3, 3])
            I = np.zeros([3, 3])
            for i in range(3):
                Theta0i.vec.FV().NumPy()[:] = Theta0Sols[:, i]
                xii = xivec[i]
                if i == 0:
                    Theta1i.vec.data = Theta1.vec.data
                if i == 1:
                    Theta1i.vec.data = Theta2.vec.data
                if i == 2:
                    Theta1i.vec.data = Theta3.vec.data
                for j in range(i + 1):
                    Theta0j.vec.FV().NumPy()[:] = Theta0Sols[:, j]
                    xij = xivec[j]
                    if j == 0:
                        Theta1j.vec.data = Theta1.vec.data
                    if j == 1:
                        Theta1j.vec.data = Theta2.vec.data
                    if j == 2:
                        Theta1j.vec.data = Theta3.vec.data

                    # Real and Imaginary parts
                    with TaskManager():
                        R[i, j] = -(((alpha ** 3) / 4) * Integrate((mu_inv) * (curl(Theta1j) * Conj(curl(Theta1i))),
                                                                   mesh, order=Integration_Order)).real
                        I[i, j] = ((alpha ** 3) / 4) * Integrate(inout * nu_no_omega * Omega * sigma * (
                                    (Theta1j + Theta0j + xij) * (Conj(Theta1i) + Theta0i + xii)), mesh,
                                                                 order=Integration_Order).real

            # Mirror tensor
            R += np.transpose(R - np.diag(np.diag(R)))
            I += np.transpose(I - np.diag(np.diag(I)))

            # Save in arrays
            TensorArray[k, :] = (N0 + R + 1j * I).flatten()
            EigenValues[k, :] = np.sort(np.linalg.eigvals(N0 + R)) + 1j * np.sort(np.linalg.eigvals(I))

        # To reduce memory usage, we delete inverse, a, and c at the end of each iteration.
        del inverse, a, c

    del enumerator
    del f1, f2, f3, ftemp
    del res
    del Theta0i, Theta1i, Theta0j, Theta1j
    del Theta3, Theta2, Theta1
    gc.collect()

    if Tensors == True and Vectors == True:
        return TensorArray, EigenValues, Theta1Sols
    elif Tensors == True:
        return TensorArray, EigenValues
    else:
        return Theta1Sols


