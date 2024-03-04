import numpy as np
# from ngsolve import *

def calc_error_certificates(Array, alphaLB, G_Store, cutoff, alpha, Sols):
    """
    James Elgy 2023
    Function to calculate error certificates for POD.
    Parameters
    ----------
    Array - Frequency Array under consideration
    alphaLB - Lower bound on stability constant
    G_Store - 6 G matrices from pg 1951 of paper. Below eqn 31.
    cutoff - Number of retained modes
    alpha - object scaling alpha.
    Sols - Ndof x N x 3 solution vectors.

    Returns
    -------
    N x 6 array of error coefficients.
    """

    rom1 = np.zeros([1 + 2 * cutoff, 1], dtype=complex)
    rom2 = np.zeros([1 + 2 * cutoff, 1], dtype=complex)
    rom3 = np.zeros([1 + 2 * cutoff, 1], dtype=complex)
    ErrorTensors = np.zeros([len(Array), 6])
    G1 = G_Store[:, :, 0]
    G2 = G_Store[:, :, 1]
    G3 = G_Store[:, :, 2]
    G12 = G_Store[:, :, 3]
    G13 = G_Store[:, :, 4]
    G23 = G_Store[:, :, 5]

    for k, omega in enumerate(Array):
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

    return ErrorTensors