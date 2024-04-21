import numpy as np
# from ngsolve import *
# import scipy.sparse as sp
# import gc
import tqdm

def Theta1_Lower_Sweep_Mat_Method(Array, Q_array, c1_array, c5_array, c7, c8_array,
                                  At0_array, At0U_array, UAt0_array, T_array, EU_array, EU_array_notconjed,
                                  Sols, G_Store, cutoff, NOF, alpha, calc_errortensors):
    """_summary_

    Args:
        Array (list): N frequencies
        Q_array, c1_array, c5_array, c7, c8_array, At0_array, At0U_array
        UAt0_array, T_array, EU_array, EU_array_notconjed:  Matrices build using the construct_matrices function.
        Sols (np.ndarray): Solution vectors
        G_Store (np.ndarray): Matrices used for computing errors
        cutoff (int): number of retained TSVD modes.
        NOF (int): Total number of frequenices
        alpha (float): object scaling
        calc_errortensors (bool): flag to compute errors

    Returns:
        np.ndarray: Nx9 complex tensor coefficients. No N0.
    """
    
    
    if calc_errortensors is True:
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

    if len(Array) == 0:
        #  we are solving for 1 frequency.
        TensorArray_no_N0 = np.zeros((1,9), dtype=complex)
    else:
        TensorArray_no_N0 = np.zeros((len(Array), 9), dtype=complex)

    for k, omega in enumerate(Array):
        nu = omega * 4*np.pi*1e-7 * (alpha ** 2)
        R = np.zeros([3, 3])
        I = np.zeros([3, 3])

        for i in range(3):
            if i == 0:
                gi = np.squeeze(Sols[:, k, 0])
            elif i == 1:
                gi = np.squeeze(Sols[:, k, 1])
            elif i == 2:
                gi = np.squeeze(Sols[:, k, 2])

            for j in range(i + 1):
                if j == 0:
                    gj = np.squeeze(Sols[:, k, 0])
                elif j == 1:
                    gj = np.squeeze(Sols[:, k, 1])
                elif j == 2:
                    gj = np.squeeze(Sols[:, k, 2])

                if i == j:
                    Q = Q_array[i]
                    T = T_array[i]
                    c1 = c1_array[i]
                    c8 = c8_array[i]
                    A_mat_t0 = At0_array[i]
                    At0U = At0U_array[i]
                    UAt0 = UAt0_array[i]
                    c5 = c5_array[i]
                    EU = EU_array[i]
                    EU_notconjed = EU_array_notconjed[i]
                elif i == 1 and j == 0:
                    Q = Q_array[3]
                    T = T_array[3]
                    c1 = c1_array[3]
                    At0U = At0U_array[3]
                    UAt0 = UAt0_array[3]
                    c8 = c8_array[3]
                    A_mat_t0 = At0_array[0]
                    c5 = c5_array[3]
                    EU = EU_array[3]
                    EU_notconjed = EU_array_notconjed[3]
                elif i == 2 and j == 0:
                    Q = Q_array[4]
                    T = T_array[4]
                    At0U = At0U_array[4]
                    UAt0 = UAt0_array[4]
                    c1 = c1_array[4]
                    c8 = c8_array[4]
                    A_mat_t0 = At0_array[0]
                    c5 = c5_array[4]
                    EU = EU_array[4]
                    EU_notconjed = EU_array_notconjed[4]
                elif i == 2 and j == 1:
                    Q = Q_array[5]
                    T = T_array[5]
                    At0U = At0U_array[5]
                    UAt0 = UAt0_array[5]
                    c1 = c1_array[5]
                    c8 = c8_array[5]
                    A_mat_t0 = At0_array[1]
                    c5 = c5_array[5]
                    EU = EU_array[5]
                    EU_notconjed = EU_array_notconjed[5]

                # Calc Real Part:
                A = np.conj(gi[None, :]) @ Q @ (gj)[:, None]
                R[i, j] = (A * (-alpha ** 3) / 4).real

                # Calc Imag Part:
                p1 = np.real(np.conj(gi) @ T @ gj)
                p2 = np.real(1 * np.conj(gj.transpose()) @  At0U)
                p2 += np.real(1 * gi.transpose() @ UAt0)
                p3 = np.real(c8 + c5)
                p4 = np.real(1 * EU @ np.conj(gj))
                p4 += np.real(1 * gi @ EU_notconjed)
                # p4 += np.real(EU.transpose() @ np.conj(gi.transpose()))

                # if omega > 1e7:
                #     print(f'{omega}')

                I[i,j] = np.real((alpha ** 3 / 4) * omega * 4*np.pi*1e-7 * alpha ** 2 * (c1 + c7[i, j] + p1 + p2 + p3 + p4))

        R += np.transpose(R - np.diag(np.diag(R))).real
        I += np.transpose(I - np.diag(np.diag(I))).real

        # Save in arrays
        TensorArray_no_N0[k,:] = (R + 1j * I).flatten()

    return TensorArray_no_N0, 0
