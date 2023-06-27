import numpy as np
from matplotlib import pyplot as plt

def Truncated_SVD(NumberofSnapshots, PODTol, Theta1Sols):
    """
    James Elgy - 2023:
    Performs truncated SVD of Theta1Sols (as Nd x Ns matrix).
    Parameters
    ----------
    NumberofSnapshots: int number of snapshots used to generate Theta1Sols
    PODTol: float trunction tolerance for SVD
    Theta1Sols: NdArray to be decomposed.

    Returns
    -------
    cutoff: int number of retained modes.
    u1Truncated: NdArray of left singular matrix for i=1
    u2Truncated: NdArray of left singular matrix for i=2
    u3Truncated: NdArray of left singular matrix for i=3
    """

    print(' performing SVD              ', end='\r')
    # Perform SVD on the solution vector matrices
    u1Truncated, s1, vh1 = np.linalg.svd(Theta1Sols[:, :, 0], full_matrices=False)
    u2Truncated, s2, vh2 = np.linalg.svd(Theta1Sols[:, :, 1], full_matrices=False)
    u3Truncated, s3, vh3 = np.linalg.svd(Theta1Sols[:, :, 2], full_matrices=False)
    # Print an update on progress
    print(' SVD complete      ')
    # scale the value of the modes
    s1norm = s1 / s1[0]
    s2norm = s2 / s2[0]
    s3norm = s3 / s3[0]
    # Decide where to truncate
    cutoff = NumberofSnapshots
    for i in range(NumberofSnapshots):
        if s1norm[i] < PODTol:
            if s2norm[i] < PODTol:
                if s3norm[i] < PODTol:
                    cutoff = i
                    break
    # Truncate the SVD matrices
    u1Truncated = u1Truncated[:, :cutoff]
    u2Truncated = u2Truncated[:, :cutoff]
    u3Truncated = u3Truncated[:, :cutoff]
    print(f' Number of retained modes = {u1Truncated.shape[1]}')
    plt.figure()
    plt.semilogy(s1norm, label='i=1')
    plt.semilogy(s2norm, label='i=2')
    plt.semilogy(s3norm, label='i=3')
    plt.xlabel('Mode')
    plt.ylabel('Normalised Singular Values')
    plt.legend()
    return cutoff, u1Truncated, u2Truncated, u3Truncated
