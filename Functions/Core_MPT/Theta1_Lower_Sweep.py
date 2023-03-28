import numpy as np
from ngsolve import *
import scipy.sparse as sp
import gc
import tqdm


def Theta1_Lower_Sweep(Array, mesh, fes, fes2, Sols, u1Truncated, u2Truncated, u3Truncated, Theta0Sols, xivec, alpha,
                       sigma, mu, inout, N0, TotalNOF, counter, PODErrorBars, alphaLB, G_Store, Order, Integration_Order, Additional_Int_Order,
                       use_integral):
    # Setup variables
    Mu0 = 4 * np.pi * 10 ** (-7)
    nu_no_omega = Mu0 * (alpha ** 2)
    NOF = len(Array)
    cutoff = len(Sols[:, 0, 0])
    Theta_0i = GridFunction(fes)
    Theta_0j = GridFunction(fes)
    Theta_1i = GridFunction(fes2)
    Theta_1j = GridFunction(fes2)
    TensorArray = np.zeros([NOF, 9], dtype=complex)
    EigenValues = np.zeros([NOF, 3], dtype=complex)

    if PODErrorBars == True:
        rom1 = np.zeros([1 + 2 * cutoff, 1], dtype=complex)
        rom2 = np.zeros([1 + 2 * cutoff, 1], dtype=complex)
        rom3 = np.zeros([1 + 2 * cutoff, 1], dtype=complex)
        TensorErrors = np.zeros([NOF, 3])
        ErrorTensors = np.zeros([NOF, 6])
        G1 = G_Store[:, :, 0]
        G2 = G_Store[:, :, 1]
        G3 = G_Store[:, :, 2]
        G12 = G_Store[:, :, 3]
        G13 = G_Store[:, :, 4]
        G23 = G_Store[:, :, 5]

    # Edit James Elgy 2022 - Calculate R and I via explicit matrix multiplication rather than integrals.
    # use_integral = False
    # Faster numerical computation is acheivable via the Theta1_Lower_Sweep_Mat_Method function, but this code has been
    # left in as a well tested fallback and for the computation of the error cert
    if use_integral is False:
        u, v = fes2.TnT()
        K = BilinearForm(fes2, symmetric=True)
        K += SymbolicBFI(inout * mu ** (-1) * curl(u) * curl(v), bonus_intorder=Additional_Int_Order)
        K += SymbolicBFI((1 - inout) * curl(u) * curl(v), bonus_intorder=Additional_Int_Order)
        K.Assemble()

        A = BilinearForm(fes2, symmetric=True)
        A += SymbolicBFI(sigma * inout * (v * u), bonus_intorder=Additional_Int_Order)
        A.Assemble()
        rows, cols, vals = A.mat.COO()
        A_mat = sp.csr_matrix((vals, (rows, cols)))

        E = np.zeros((3, fes2.ndof), dtype=complex)
        G = np.zeros((3, 3))

        for i in range(3):

            E_lf = LinearForm(fes2)
            E_lf += SymbolicLFI(sigma * inout * xivec[i] * v, bonus_intorder=Additional_Int_Order)
            E_lf.Assemble()
            E[i, :] = E_lf.vec.FV().NumPy()[:]

            for j in range(3):
                G[i, j] = Integrate(sigma * inout * xivec[i] * xivec[j], mesh, order=Integration_Order)

            H = E.transpose()

        rows, cols, vals = K.mat.COO()
        Q = sp.csr_matrix((vals, (rows, cols)))
        del E_lf
        del A
        del K

        # For faster computation of tensor coefficients, we multiply with Ui before the loop.
        Q11 = np.conj(np.transpose(u1Truncated)) @ Q @ u1Truncated
        Q22 = np.conj(np.transpose(u2Truncated)) @ Q @ u2Truncated
        Q33 = np.conj(np.transpose(u3Truncated)) @ Q @ u3Truncated
        Q21 = np.conj(np.transpose(u2Truncated)) @ Q @ u1Truncated
        Q31 = np.conj(np.transpose(u3Truncated)) @ Q @ u1Truncated
        Q32 = np.conj(np.transpose(u3Truncated)) @ Q @ u2Truncated

        # Similarly for the imaginary part, we multiply with the theta0 sols beforehand.
        A_mat_t0_1 = (A_mat) @ Theta0Sols[:, 0]
        A_mat_t0_2 = (A_mat) @ Theta0Sols[:, 1]
        A_mat_t0_3 = (A_mat) @ Theta0Sols[:, 2]

        c1_11 = (np.transpose(Theta0Sols[:, 0])) @ A_mat_t0_1
        c1_22 = (np.transpose(Theta0Sols[:, 1])) @ A_mat_t0_2
        c1_33 = (np.transpose(Theta0Sols[:, 2])) @ A_mat_t0_3
        c1_21 = (np.transpose(Theta0Sols[:, 1])) @ A_mat_t0_1
        c1_31 = (np.transpose(Theta0Sols[:, 2])) @ A_mat_t0_1
        c1_32 = (np.transpose(Theta0Sols[:, 2])) @ A_mat_t0_2

        c5_11 = E[0, :] @ Theta0Sols[:, 0]
        c5_22 = E[1, :] @ Theta0Sols[:, 1]
        c5_33 = E[2, :] @ Theta0Sols[:, 2]
        c5_21 = E[1, :] @ Theta0Sols[:, 0]
        c5_31 = E[2, :] @ Theta0Sols[:, 0]
        c5_32 = E[2, :] @ Theta0Sols[:, 1]

        T11 = np.conj(np.transpose(u1Truncated)) @ A_mat @ u1Truncated
        T22 = np.conj(np.transpose(u2Truncated)) @ A_mat @ u2Truncated
        T33 = np.conj(np.transpose(u3Truncated)) @ A_mat @ u3Truncated
        T21 = np.conj(np.transpose(u2Truncated)) @ A_mat @ u1Truncated
        T31 = np.conj(np.transpose(u3Truncated)) @ A_mat @ u1Truncated
        T32 = np.conj(np.transpose(u3Truncated)) @ A_mat @ u2Truncated

    for k, omega in enumerate(tqdm.tqdm(Array, desc='Solving Reduced Order System')):

        # # This part is for obtaining the solutions in the lower dimensional space
        # try:
        #     counter.value += 1
        #     print(' solving reduced order system %d/%d    ' % (counter.value, TotalNOF), end='\r')
        # except:
        #     print(' solving reduced order system', end='\r')

        # This part projects the problem to the higher dimensional space
        W1 = np.dot(u1Truncated, Sols[:, k, 0]).flatten()
        W2 = np.dot(u2Truncated, Sols[:, k, 1]).flatten()
        W3 = np.dot(u3Truncated, Sols[:, k, 2]).flatten()

        # Calculate the tensors
        nu = omega * Mu0 * (alpha ** 2)
        R = np.zeros([3, 3])
        I = np.zeros([3, 3])

        if use_integral is True:
            for i in range(3):
                Theta_0i.vec.FV().NumPy()[:] = Theta0Sols[:, i]
                xii = xivec[i]
                if i == 0:
                    Theta_1i.vec.FV().NumPy()[:] = W1
                if i == 1:
                    Theta_1i.vec.FV().NumPy()[:] = W2
                if i == 2:
                    Theta_1i.vec.FV().NumPy()[:] = W3
                for j in range(i + 1):
                    Theta_0j.vec.FV().NumPy()[:] = Theta0Sols[:, j]
                    xij = xivec[j]
                    if j == 0:
                        Theta_1j.vec.FV().NumPy()[:] = W1
                    if j == 1:
                        Theta_1j.vec.FV().NumPy()[:] = W2
                    if j == 2:
                        Theta_1j.vec.FV().NumPy()[:] = W3

                    # Real and Imaginary parts
                    R[i, j] = -(((alpha ** 3) / 4) * Integrate((mu ** (-1)) * (curl(Theta_1j) * Conj(curl(Theta_1i))),
                                                               mesh, order=Integration_Order)).real
                    I[i, j] = ((alpha ** 3) / 4) * Integrate(
                        inout * nu * sigma * ((Theta_1j + Theta_0j + xij) * (Conj(Theta_1i) + Theta_0i + xii)), mesh,
                        order=Integration_Order).real

        # Use matrix method.
        else:
            for i in range(3):
                for i in range(3):
                    t0i = Theta0Sols[:, i] + 1j * np.zeros(Theta0Sols[:, i].shape)
                    if i == 0:
                        gi = np.squeeze(Sols[:, k, 0])
                        wi = W1
                    elif i == 1:
                        gi = np.squeeze(Sols[:, k, 1])
                        wi = W2
                    elif i == 2:
                        gi = np.squeeze(Sols[:, k, 2])
                        wi = W3

                    for j in range(i + 1):
                        t0j = Theta0Sols[:, j] + 1j * np.zeros(Theta0Sols[:, j].shape)
                        if j == 0:
                            gj = np.squeeze(Sols[:, k, 0])
                            wj = W1
                        elif j == 1:
                            gj = np.squeeze(Sols[:, k, 1])
                            wj = W2
                        elif j == 2:
                            gj = np.squeeze(Sols[:, k, 2])
                            wj = W3

                        if i == 0 and j == 0:
                            Q = Q11
                            T = T11
                            c1 = c1_11
                            A_mat_t0 = A_mat_t0_1
                            c5 = c5_11
                        elif i == 1 and j == 1:
                            Q = Q22
                            T = T22
                            c1 = c1_22
                            A_mat_t0 = A_mat_t0_2
                            c5 = c5_22
                        elif i == 2 and j == 2:
                            Q = Q33
                            c1 = c1_33
                            T = T33
                            A_mat_t0 = A_mat_t0_3
                            c5 = c5_33

                        elif i == 1 and j == 0:
                            Q = Q21
                            T = T21
                            c1 = c1_21
                            A_mat_t0 = A_mat_t0_1
                            c5 = c5_21
                        elif i == 2 and j == 0:
                            Q = Q31
                            T = T31
                            c1 = c1_31
                            A_mat_t0 = A_mat_t0_1
                            c5 = c5_31
                        elif i == 2 and j == 1:
                            Q = Q32
                            T = T32
                            c1 = c1_32
                            A_mat_t0 = A_mat_t0_2
                            c5 = c5_32

                        A = np.conj(gi[None, :]) @ Q @ (gj)[:, None]
                        R[i, j] = (A * (-alpha ** 3) / 4).real

                        c2 = wi[None, :] @ A_mat_t0
                        c3 = (t0i)[None, :] @ A_mat @ np.conj(wj)[:, None]
                        c4 = (wi)[None, :] @ A_mat @ np.conj(wj)[:, None]
                        c6 = (E[i, :]) @ np.conj(wj)[:, None]
                        c8 = t0i[None, :] @ H[:, j]
                        c9 = wi[None, :] @ H[:, j]

                        c_sum = np.real(c2 + c3 + c4 + c5 + c6 + c8 + c9)

                        I[i, j] = np.real((alpha ** 3 / 4) * omega * Mu0 * alpha ** 2 * (c1 + G[i, j] + c_sum))

        R += np.transpose(R - np.diag(np.diag(R))).real
        I += np.transpose(I - np.diag(np.diag(I))).real

        # Save in arrays
        TensorArray[k, :] = (N0 + R + 1j * I).flatten()
        EigenValues[k, :] = np.sort(np.linalg.eigvals(N0 + R)) + 1j * np.sort(np.linalg.eigvals(I))

        if PODErrorBars == True:
            rom1[0, 0] = omega
            rom2[0, 0] = omega
            rom3[0, 0] = omega

            rom1[1:1 + cutoff, 0] = -Sols[:, k, 0].flatten()
            rom2[1:1 + cutoff, 0] = -Sols[:, k, 1].flatten()
            rom3[1:1 + cutoff, 0] = -Sols[:, k, 2].flatten()

            rom1[1 + cutoff:, 0] = -(Sols[:, k, 0] * omega).flatten()
            rom2[1 + cutoff:, 0] = -(Sols[:, k, 1] * omega).flatten()
            rom3[1 + cutoff:, 0] = -(Sols[:, k, 2] * omega).flatten()

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
                    ErrorTensors[k, j] = ((alpha ** 3) / 4) * (Errors[j] ** 2) / alphaLB
                else:
                    ErrorTensors[k, j] = -2 * Errors[j]
                    if j == 3:
                        ErrorTensors[k, j] += (Errors[0] ** 2) + (Errors[1] ** 2)
                        ErrorTensors[k, j] = ((alpha ** 3) / (8 * alphaLB)) * (
                                    (Errors[0] ** 2) + (Errors[1] ** 2) + ErrorTensors[k, j])
                    if j == 4:
                        ErrorTensors[k, j] += (Errors[0] ** 2) + (Errors[2] ** 2)
                        ErrorTensors[k, j] = ((alpha ** 3) / (8 * alphaLB)) * (
                                    (Errors[0] ** 2) + (Errors[1] ** 2) + ErrorTensors[k, j])
                    if j == 5:
                        ErrorTensors[k, j] += (Errors[1] ** 2) + (Errors[2] ** 2)
                        ErrorTensors[k, j] = ((alpha ** 3) / (8 * alphaLB)) * (
                                    (Errors[0] ** 2) + (Errors[1] ** 2) + ErrorTensors[k, j])

    if use_integral is False:
        del Q, Q11, Q22, Q33, Q21, Q31, Q32
        del T, T11, T22, T33, T21, T31, T32
        del c1_11, c1_22, c1_33, c1_21, c1_31, c1_32
        del c5_11, c5_22, c5_33, c5_21, c5_31, c5_32
    del Theta_0i, Theta_1i, Theta_0j, Theta_1j
    gc.collect()

    if PODErrorBars == True:
        return TensorArray, EigenValues, ErrorTensors
    else:
        return TensorArray, EigenValues

