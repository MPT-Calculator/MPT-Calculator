import numpy as np
from ngsolve import *
import scipy.sparse as sp
import gc

def Mat_Method_Calc_Real_Part(bilinear_bonus_int_order: int,
                              fes2: comp.HCurl,
                              inout: fem.CoefficientFunction,
                              mu_inv: comp.GridFunction,
                              alpha: float,
                              Sols: np.ndarray,
                              u1Truncated,
                              u2Truncated,
                              u3Truncated,
                              NumSolverThreads: int | str,
                              drop_tol: float | None,
                              BigProblem: bool,
                              ReducedSolve=True) -> np.ndarray:
    """
    James Elgy - 2024.
    Function to compute the real tensor coefficients (R)_ij efficiently using the faster matrix method.
    
    1) Computes the bilinear form K
    2) If reduced solve is True, reduce K to size MxM
    3) For each frequency, compute conj(q_i)^T Q_ij (q_j)
    4) Scale and compute (R)_ij
    
    If BigProblem is True, then a slower but more memory efficient implementation is used using K.Apply().
    
    Args:
        bilinear_bonus_int_order (int): Integration order for the bilinear forms
        fes2 (comp.HCurl): HCurl finite element space for the Theta1 problem.
        inout (fem.CoefficientFunction): material coefficient function. 1 inside objects, 0 outside
        mu_inv (comp.GridFunction): grid function for 1/mu_r. Note that for material discontinuities aligning with vertices no interpolation is done.
        alpha (float): object size scaling
        Sols (np.ndarray): Ndof x nfreqs x 3 vector of solution coefficients.
        u1Truncated (_type_): Ndof x M complex left singular matrix for e_1. If ReducedSolve is False, then replace U with sparse identity of size Ndof.
        u2Truncated (_type_): Ndof x M complex left singular mactrix for e_2. If ReducedSolve is False, then replace U with sparse identity of size Ndof.
        u3Truncated (_type_): Ndof x M complex left singular matrix for e_3. If ReducedSolve is False, then replace U with sparse identity of size Ndof.
        NumSolverThreads (int | str): Multithreading threads. If using all threads use 'default'.
        drop_tol (float | None): During assembly entries < drop_tol are assumed to be 0. Use None to include all entries.
        BigProblem (bool): if True then the code does not assemble the system matrix entirely. Slower but more memory efficient.
        ReducedSolve (bool, optional): If True, the size of the multiplications are reduced to size M. Use with POD. Defaults to True.

    Returns:
        np.ndarray: Nfreq x 9 array of real tensor coeffcients.
    """
    
    if NumSolverThreads != 'default':
        SetNumThreads(NumSolverThreads)
    
    u, v = fes2.TnT()
    ndof2 = fes2.ndof
    
    if ReducedSolve is True:
        cutoff = u1Truncated.shape[1]
    
    
    K = BilinearForm(fes2, symmetric=True, delete_zero_elements =drop_tol,keep_internal=False, symmetric_storage=True, nonassemble = BigProblem)
    K += SymbolicBFI(inout * mu_inv * curl(u) * Conj(curl(v)), bonus_intorder=bilinear_bonus_int_order)
    K += SymbolicBFI((1 - inout) * curl(u) * Conj(curl(v)), bonus_intorder=bilinear_bonus_int_order)
    
    if BigProblem is False:
        with TaskManager():
            K.Assemble()
        rows, cols, vals = K.mat.COO()
        del K
        Qsym = sp.csr_matrix((vals, (rows, cols)),shape=(fes2.ndof,fes2.ndof))
        Q = Qsym + Qsym.T - sp.diags(Qsym.diagonal())
        del Qsym
        del rows, cols, vals
        gc.collect()
    
    # if ReducedSolve is False then the Q matrix is not shrunk and there is no point in multiplying it with anything.
    if BigProblem is False:
        # Reducing size of K matrix
        if ReducedSolve is True:
            Q11 = np.conj(np.transpose(u1Truncated)) @ Q @ u1Truncated
            Q22 = np.conj(np.transpose(u2Truncated)) @ Q @ u2Truncated
            Q33 = np.conj(np.transpose(u3Truncated)) @ Q @ u3Truncated
            Q21 = np.conj(np.transpose(u2Truncated)) @ Q @ u1Truncated
            Q31 = np.conj(np.transpose(u3Truncated)) @ Q @ u1Truncated
            Q32 = np.conj(np.transpose(u3Truncated)) @ Q @ u2Truncated
            
        else:
            Q11 = Q22 = Q33 = Q21 = Q31 = Q32 = Q
    else:
        if ReducedSolve is True:
            # Reducing size of K matrix
            QU1 = np.zeros([ndof2, cutoff], dtype=complex)
            read_vec = GridFunction(fes2).vec.CreateVector()
            write_vec = GridFunction(fes2).vec.CreateVector()
            
            # For each column in u1Truncated, post multiply with K. We then premultiply by appropriate vector to reduce size to MxM.
            for i in range(cutoff):
                read_vec.FV().NumPy()[:] = u1Truncated[:, i]
                with TaskManager():
                    K.Apply(read_vec, write_vec)
                QU1[:, i] = write_vec.FV().NumPy()
            Q11 = np.conj(np.transpose(u1Truncated)) @ QU1
            Q21 = np.conj(np.transpose(u2Truncated)) @ QU1
            Q31 = np.conj(np.transpose(u3Truncated)) @ QU1
            del QU1
            
            # Same as before
            QU2 = np.zeros([ndof2, cutoff], dtype=complex)
            for i in range(cutoff):
                read_vec.FV().NumPy()[:] = u2Truncated[:, i]
                with TaskManager():
                    K.Apply(read_vec, write_vec)
                QU2[:, i] = write_vec.FV().NumPy()
            Q22 = np.conj(np.transpose(u2Truncated)) @ QU2
            Q32 = np.conj(np.transpose(u3Truncated)) @ QU2
            del QU2
            
            # Same as before.
            QU3 = np.zeros([ndof2, cutoff], dtype=complex)
            for i in range(cutoff):
                read_vec.FV().NumPy()[:] = u3Truncated[:, i]
                with TaskManager():
                    K.Apply(read_vec, write_vec)
                QU3[:, i] = write_vec.FV().NumPy()
            Q33 = np.conj(np.transpose(u3Truncated)) @ QU3

                
    # Computing Tensor Coefficients:

    real_part = np.zeros((Sols.shape[1], 9))
    
    # For each frequency pre and post multiply Q with the solution vector for i=0:3, j=0:i+1.
    for k in range(Sols.shape[1]):
        
        print(f'{k} / {Sols.shape[1]}', end='\r')
        
        R = np.zeros([3, 3])
        
        # if ReducedSolve is True then the matrices are small so we can just multiply them.
        if ReducedSolve is True or BigProblem is False:
            for i in range(3):
                gi = np.squeeze(Sols[:,k, i])
                for j in range(i + 1):
                    gj = np.squeeze(Sols[:, k, j])

                    Qij = locals()[f'Q{i+1}{j+1}']

                    A = np.conj(gi[None, :]) @ Qij @ (gj)[:, None]
                    R[i, j] = (A * (-alpha ** 3) / 4).real

        # If we don't reduce the size of the matrix and we still want to save memory, then we can use K.Apply here as well.
        elif BigProblem is True and ReducedSolve is False:
            read_vec = GridFunction(fes2).vec.CreateVector()
            write_vec = GridFunction(fes2).vec.CreateVector()
            for i in range(3):
                gi = np.squeeze(Sols[:,k,i])
                for j in range(i + 1):
                    gj = np.squeeze(Sols[:, k, j])
                    
                    read_vec.FV().NumPy()[:] = gj
                    with TaskManager():
                        K.Apply(read_vec, write_vec)
                        
                    A = np.conj(gi[None,:]) @ write_vec.FV().NumPy()
                    R[i, j] = (A * (-alpha ** 3) / 4).real
                    
        
        R += np.transpose(R - np.diag(np.diag(R))).real
        real_part[k,:] = R.flatten()
    
    return real_part