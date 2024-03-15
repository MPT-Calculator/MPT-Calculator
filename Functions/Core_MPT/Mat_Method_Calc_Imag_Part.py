import numpy as np
from ngsolve import *
import scipy.sparse as sp
import gc

def Mat_Method_Calc_Imag_Part(Array: np.ndarray, 
                              Integration_Order: int,
                              Theta0Sol: np.ndarray,
                              bilinear_bonus_int_order: int,
                              fes2: comp.HCurl,
                              mesh: comp.Mesh,
                              inout: fem.CoefficientFunction,
                              alpha: float,
                              Sols: np.ndarray,
                              sigma: comp.GridFunction,
                              u1Truncated,
                              u2Truncated,
                              u3Truncated,
                              xivec: list,
                              NumSolverThreads: int | str,
                              drop_tol: float | None,
                              BigProblem: bool,
                              ReducedSolve=True) -> np.ndarray:
    
    """
    James Elgy - 2024.
    Function to compute the imag tensor coefficients (I)_ij efficiently using the faster matrix method.
    
    1) Computes the bilinear form A
    2) Computes matrices E, G, and H.
    2) If reduced solve is True, reduce A to size MxM and E and H to size 3xM.
    4) Compute additional matrices and vectors (𝐨ⱼ)ᵀ (̅𝐂²)ᴹ,  (𝐨ⱼ)ᵀ (𝐂²)ᴹ,  𝐨ⱼᵀ 𝐂⁽¹⁾ 𝐨ᵢ, 𝐬ᵢᵀ 𝐨ⱼ, 𝐬ⱼᵀ 𝐨ᵢ, and (𝐭ᴹ)ᵀ.
    3) For each frequency, compute conj(q_i)^T A_ij (q_j)
    4) Scale and compute (I)_ij
    
    If BigProblem is True, then a slower but more memory efficient implementation is used using A.Apply().
    
    Args:
        Array (np.ndarray): Array of frequencies to consider.
        Integration order (int): order to use for integration in Integrate function.
        Theta0Sol (np.ndarray): ndof x 3 array of theta0 solutions.
        bilinear_bonus_int_order (int): Integration order for the bilinear forms
        fes2 (comp.HCurl): HCurl finite element space for the Theta1 problem.
        mesh (comp.Mesh): ngsolve mesh.
        inout (fem.CoefficientFunction): material coefficient function. 1 inside objects, 0 outside
        alpha (float): object size scaling
        Sols (np.ndarray): Ndof x nfreqs x 3 vector of solution coefficients.
        sigma (comp.GridFunction): Grid Function for sigma. Note that for material discontinuities aligning with vertices no interpolation is done
        u1Truncated (_type_): Ndof x M complex left singular matrix for e_1. If ReducedSolve is False, then replace U with sparse identity of size Ndof.
        u2Truncated (_type_): Ndof x M complex left singular mactrix for e_2. If ReducedSolve is False, then replace U with sparse identity of size Ndof.
        u3Truncated (_type_): Ndof x M complex left singular matrix for e_3. If ReducedSolve is False, then replace U with sparse identity of size Ndof.
        xivec (list): 3x3 list of direction vectors
        NumSolverThreads (int | str): Multithreading threads. If using all threads use 'default'.
        drop_tol (float | None): During assembly entries < drop_tol are assumed to be 0. Use None to include all entries.
        BigProblem (bool): if True then the code does not assemble the system matrix entirely. Slower but more memory efficient.
        ReducedSolve (bool, optional): If True, the size of the multiplications are reduced to size M. Use with POD. Defaults to True.

    Returns:
        np.ndarray: Nfreq x 9 array of imag tensor coeffcients.
    """

    
    if NumSolverThreads != 'default':
        SetNumThreads(NumSolverThreads)
    
    u, v = fes2.TnT()
    ndof2 = fes2.ndof
    
    if ReducedSolve is True:
        cutoff = u1Truncated.shape[1]
    
    A = BilinearForm(fes2, symmetric=True, delete_zero_elements =drop_tol,keep_internal=False, symmetric_storage=True, nonassemble = BigProblem)
    A += SymbolicBFI(sigma * inout * (v * u), bonus_intorder=bilinear_bonus_int_order)
    
    if BigProblem is False:
        with TaskManager():
            A.Assemble()
        rows, cols, vals = A.mat.COO()
        del A
        A_matsym = sp.csr_matrix((vals, (rows, cols)),shape=(fes2.ndof,fes2.ndof))
        del rows, cols, vals
        A_mat = A_matsym + A_matsym.T - sp.diags(A_matsym.diagonal())
        del A_matsym
        gc.collect()
    
    # E and G are both small compared to A so storing them shouldn't be an issue.
    E = np.zeros((3, fes2.ndof), dtype=complex)
    G = np.zeros((3, 3))
    
    for i in range(3):
        E_lf = LinearForm(fes2)
        E_lf += SymbolicLFI(sigma * inout * xivec[i] * v, bonus_intorder=bilinear_bonus_int_order)
        E_lf.Assemble()
        E[i, :] = E_lf.vec.FV().NumPy()[:]
        del E_lf
        
        for j in range(3):
            G[i, j] = Integrate(sigma * inout * xivec[i] * xivec[j], mesh, order=Integration_Order)
    H = E.transpose()
    
    
    if BigProblem is False:
        A_mat_t0_1 = (A_mat) @ Theta0Sol[:, 0]
        A_mat_t0_2 = (A_mat) @ Theta0Sol[:, 1]
        A_mat_t0_3 = (A_mat) @ Theta0Sol[:, 2]
    else:
        A_mat_t0_1 = np.zeros(ndof2, dtype=complex)
        A_mat_t0_2 = np.zeros(ndof2, dtype=complex)
        A_mat_t0_3 = np.zeros(ndof2, dtype=complex)
        read_vec = GridFunction(fes2).vec.CreateVector()
        write_vec = GridFunction(fes2).vec.CreateVector()
        
        read_vec.FV().NumPy()[:] = Theta0Sol[:, 0]
        with TaskManager():
            A.Apply(read_vec, write_vec)
        A_mat_t0_1[:] = write_vec.FV().NumPy()
        
        read_vec = GridFunction(fes2).vec.CreateVector()
        write_vec = GridFunction(fes2).vec.CreateVector()
        
        read_vec.FV().NumPy()[:] = Theta0Sol[:, 1]
        with TaskManager():
            A.Apply(read_vec, write_vec)
        A_mat_t0_2[:] = write_vec.FV().NumPy()
        
        read_vec = GridFunction(fes2).vec.CreateVector()
        write_vec = GridFunction(fes2).vec.CreateVector()
        
        read_vec.FV().NumPy()[:] = Theta0Sol[:, 2]
        with TaskManager():
            A.Apply(read_vec, write_vec)
        A_mat_t0_3[:] = write_vec.FV().NumPy()
    
    
    # if ReducedSolve is False then the A matrix is not shrunk and there is no point in multiplying it with anything.
    if BigProblem is False:
        # (𝐂)^M being the reduced MxM complex matrix. Similarly to the real part, we store each combination of i,j.
        if ReducedSolve == True:
            T11 = np.conj(np.transpose(u1Truncated)) @ A_mat @ u1Truncated
            T22 = np.conj(np.transpose(u2Truncated)) @ A_mat @ u2Truncated
            T33 = np.conj(np.transpose(u3Truncated)) @ A_mat @ u3Truncated
            T21 = np.conj(np.transpose(u2Truncated)) @ A_mat @ u1Truncated
            T31 = np.conj(np.transpose(u3Truncated)) @ A_mat @ u1Truncated
            T32 = np.conj(np.transpose(u3Truncated)) @ A_mat @ u2Truncated
        else:
            T11 = T22 = T33 = T21 = T31 = T32 = A_mat
            
    else:
        if ReducedSolve is True:
            # Reducing size of K matrix
            TU1 = np.zeros([ndof2, cutoff], dtype=complex)
            read_vec = GridFunction(fes2).vec.CreateVector()
            write_vec = GridFunction(fes2).vec.CreateVector()
            
            # For each column in u1Truncated, post multiply with K. We then premultiply by appropriate vector to reduce size to MxM.
            for i in range(cutoff):
                read_vec.FV().NumPy()[:] = u1Truncated[:, i]
                with TaskManager():
                    A.Apply(read_vec, write_vec)
                TU1[:, i] = write_vec.FV().NumPy()
            T11 = np.conj(np.transpose(u1Truncated)) @ TU1
            T21 = np.conj(np.transpose(u2Truncated)) @ TU1
            T31 = np.conj(np.transpose(u3Truncated)) @ TU1
            del TU1
            
            # Same as before
            TU2 = np.zeros([ndof2, cutoff], dtype=complex)
            for i in range(cutoff):
                read_vec.FV().NumPy()[:] = u2Truncated[:, i]
                with TaskManager():
                    A.Apply(read_vec, write_vec)
                TU2[:, i] = write_vec.FV().NumPy()
            T22 = np.conj(np.transpose(u2Truncated)) @ TU2
            T32 = np.conj(np.transpose(u3Truncated)) @ TU2
            del TU2
            
            # Same as before.
            TU3 = np.zeros([ndof2, cutoff], dtype=complex)
            for i in range(cutoff):
                read_vec.FV().NumPy()[:] = u3Truncated[:, i]
                with TaskManager():
                    A.Apply(read_vec, write_vec)
                TU3[:, i] = write_vec.FV().NumPy()
            T33 = np.conj(np.transpose(u3Truncated)) @ TU3
            del TU3
        
    # At this stage, all the work relating to the large bilinear form A has been completed. All the remaining matrix multiplications
    # concern smaller matrices and so BigProblem is no longer considered.
    
    if BigProblem is False:
        # At this point, we have constructed each of the main matrices we need and obtained the reduced A matrix. The
        # larger bilinear form can therefore be removed to save memory.
        del A_mat
    
    
    # At0_array = [A_mat_t0_1, A_mat_t0_2, A_mat_t0_3]
    # Here we compute (𝐨ⱼ)ᵀ (̅𝐂²)ᴹ
    # Renamed to better fit naming convention
    if ReducedSolve == True:
        UAt011_conj = np.conj(u1Truncated.transpose()) @ A_mat_t0_1
        UAt022_conj = np.conj(u2Truncated.transpose()) @ A_mat_t0_2
        UAt033_conj = np.conj(u3Truncated.transpose()) @ A_mat_t0_3
        UAt012_conj = np.conj(u1Truncated.transpose()) @ A_mat_t0_2
        UAt013_conj = np.conj(u1Truncated.transpose()) @ A_mat_t0_3
        UAt023_conj = np.conj(u2Truncated.transpose()) @ A_mat_t0_3
    else:
        UAt011_conj = A_mat_t0_1
        UAt022_conj = UAt012_conj = A_mat_t0_2
        UAt033_conj =  UAt013_conj = UAt023_conj = A_mat_t0_3
        
    # UAt0_conj = [UAt011_conj, UAt022_conj, UAt033_conj, UAt012_conj, UAt013_conj, UAt023_conj]
    # Similarly we compute and store (𝐨ⱼ)ᵀ (𝐂²)ᴹ
    if ReducedSolve == True:
        UAt011 = (u1Truncated.transpose()) @ A_mat_t0_1
        UAt022 = (u2Truncated.transpose()) @ A_mat_t0_2
        UAt033 = (u3Truncated.transpose()) @ A_mat_t0_3
        UAt021 = (u2Truncated.transpose()) @ A_mat_t0_1
        UAt031 = (u3Truncated.transpose()) @ A_mat_t0_1
        UAt032 = (u3Truncated.transpose()) @ A_mat_t0_2
    else:
        UAt011 = UAt021 = UAt031 = A_mat_t0_1
        UAt022 = UAt032 = A_mat_t0_2
        UAt033 = A_mat_t0_3
    
    # UAt0U_array = [UAt011, UAt022, UAt033, UAt021, UAt031, UAt032]
    # Finally, we can construct constants that do not depend on frequency.
    # the constant c1 corresponds to 𝐨ⱼᵀ 𝐂⁽¹⁾ 𝐨ᵢ. Similar to other cases we store each combination of i and j.
    c1_11 = (np.transpose(Theta0Sol[:, 0])) @ A_mat_t0_1
    c1_22 = (np.transpose(Theta0Sol[:, 1])) @ A_mat_t0_2
    c1_33 = (np.transpose(Theta0Sol[:, 2])) @ A_mat_t0_3
    c1_21 = (np.transpose(Theta0Sol[:, 1])) @ A_mat_t0_1
    c1_31 = (np.transpose(Theta0Sol[:, 2])) @ A_mat_t0_1
    c1_32 = (np.transpose(Theta0Sol[:, 2])) @ A_mat_t0_2
    # c5 corresponds to 𝐬ᵢᵀ 𝐨ⱼ. Note that E has been transposed here.
    c5_11 = E[0, :] @ Theta0Sol[:, 0]
    c5_22 = E[1, :] @ Theta0Sol[:, 1]
    c5_33 = E[2, :] @ Theta0Sol[:, 2]
    c5_21 = E[1, :] @ Theta0Sol[:, 0]
    c5_31 = E[2, :] @ Theta0Sol[:, 0]
    c5_32 = E[2, :] @ Theta0Sol[:, 1]
    # Similarly to other examples we store each combination rather than recompute
    # c1_array = [c1_11, c1_22, c1_33, c1_21, c1_31, c1_32]
    # c5_array = [c5_11, c5_22, c5_33, c5_21, c5_31, c5_32]
    # c7 = G corresponds to cᵢⱼ from paper. Note that G does not depend on the FEM basis functions, rather is a
    # polynomial.
    c7 = G
    # c8 corresponds to  𝐬ⱼᵀ 𝐨ᵢ and shold equal c5 for on diagonal entries.
    c8_11 = Theta0Sol[:, 0] @ H[:, 0]
    c8_22 = Theta0Sol[:, 1] @ H[:, 1]
    c8_33 = Theta0Sol[:, 2] @ H[:, 2]
    c8_21 = Theta0Sol[:, 1] @ H[:, 0]
    c8_31 = Theta0Sol[:, 2] @ H[:, 0]
    c8_32 = Theta0Sol[:, 2] @ H[:, 1]
    # c8_array = [c8_11, c8_22, c8_33, c8_21, c8_31, c8_32]
    # EU is the reduced linear form for E. Here we compute (̅𝐭ᴹ)ᵀ.
    if ReducedSolve == True:
        EU_11 = E[0, :] @ np.conj(u1Truncated)
        EU_22 = E[1, :] @ np.conj(u2Truncated)
        EU_33 = E[2, :] @ np.conj(u3Truncated)
        EU_21 = E[1, :] @ np.conj(u1Truncated)
        EU_31 = E[2, :] @ np.conj(u1Truncated)
        EU_32 = E[2, :] @ np.conj(u2Truncated)
    else:
        EU_11 = E[0, :]
        EU_22 = EU_21 = E[1, :]
        EU_33 = EU_31 = EU_32 = E[2, :]
    # EU_array_conj = [EU_11, EU_22, EU_33, EU_21, EU_31, EU_32]
    H = E.transpose()
    # also computing  (𝐭ᴹ)ᵀ
    # Renamed to better fit naming convention
    if ReducedSolve == True:
        UH_11 = u1Truncated.transpose() @ H[:, 0]
        UH_22 = u2Truncated.transpose() @ H[:, 1]
        UH_33 = u3Truncated.transpose() @ H[:, 2]
        UH_21 = u2Truncated.transpose() @ H[:, 0]
        UH_31 = u3Truncated.transpose() @ H[:, 0]
        UH_32 = u3Truncated.transpose() @ H[:, 1]
    else:
        UH_11 = UH_21 = UH_31 =  H[:, 0]
        UH_22 = UH_32 =  H[:, 1]
        UH_33 = H[:, 2]
        
    # UH_array = [UH_11, UH_22, UH_33, UH_21, UH_31, UH_32]
    
    # At this point, we have constrcted all of the matrices, vectors, and constants that need to be computed.
    # We can now iterate through frequency and compute the tensor coefficients.
    
    # Computing Tensor Coefficients:
    
    imag_part = np.zeros((Sols.shape[1], 9))
    
    # For each frequency pre and post multiply Q with the solution vector for i=0:3, j=0:i+1.
    for k, omega in enumerate(Array):
        
        print(f'{k} / {len(Array)}', end='\r')
        
        I = np.zeros([3, 3])
        if ReducedSolve is True or BigProblem is False:
            for i in range(3):
                gi = np.squeeze(Sols[:,k,i])
                for j in range(i + 1):
                    gj = np.squeeze(Sols[:, k, j])
                    
                    UH = locals()[f'UH_{i+1}{j+1}']
                    EU = locals()[f'EU_{i+1}{j+1}']
                    T = locals()[f'T{i+1}{j+1}']
                    c1 = locals()[f'c1_{i+1}{j+1}']
                    c8 = locals()[f'c8_{i+1}{j+1}']
                    c5 = locals()[f'c5_{i+1}{j+1}']
                    UAt0 = locals()[f'UAt0{i+1}{j+1}']
                    At0U = locals()[f'UAt0{j+1}{i+1}_conj']
                    
                    # Calc Imag Part:
                    p1 = np.real(np.conj(gi) @ T @ gj)
                    p2 = np.real(1 * np.conj(gj.transpose()) @  At0U)
                    p2 += np.real(1 * gi.transpose() @ UAt0)
                    p3 = np.real(c8 + c5)
                    p4 = np.real(1 * EU @ np.conj(gj))
                    p4 += np.real(1 * gi @ UH)

                    I[i,j] = np.real((alpha ** 3 / 4) * omega * 4*np.pi*1e-7 * alpha ** 2 * (c1 + c7[i, j] + p1 + p2 + p3 + p4))
        
        # If we don't reduce the size of the matrix and we still want to save memory, then we can use K.Apply here as well.
        elif BigProblem is True:
            
            for i in range(3):
                gi = np.squeeze(Sols[:,k,i])
                for j in range(i + 1):
                    gj = np.squeeze(Sols[:, k, j])
                    UH = locals()[f'UH_{i+1}{j+1}']
                    EU = locals()[f'EU_{i+1}{j+1}']
                    # T = locals()[f'T{i+1}{j+1}']
                    c1 = locals()[f'c1_{i+1}{j+1}']
                    c8 = locals()[f'c8_{i+1}{j+1}']
                    c5 = locals()[f'c5_{i+1}{j+1}']
                    UAt0 = locals()[f'UAt0{i+1}{j+1}']
                    At0U = locals()[f'UAt0{j+1}{i+1}_conj']
                    
                    read_vec.FV().NumPy()[:] = gj
                    with TaskManager():
                        A.Apply(read_vec, write_vec)
                    p1 = np.conj(gi) @ write_vec.FV().NumPy()[:]
                    
                    # Calc Imag Part:
                    # p1 = np.real(np.conj(gi) @ T @ gj)
                    p2 = np.real(1 * np.conj(gj.transpose()) @  At0U)
                    p2 += np.real(1 * gi.transpose() @ UAt0)
                    p3 = np.real(c8 + c5)
                    p4 = np.real(1 * EU @ np.conj(gj))
                    p4 += np.real(1 * gi @ UH)

                    I[i,j] = np.real((alpha ** 3 / 4) * omega * 4*np.pi*1e-7 * alpha ** 2 * (c1 + c7[i, j] + p1 + p2 + p3 + p4))

                    
                
                 
        I += np.transpose(I - np.diag(np.diag(I))).real
        imag_part[k,:] = I.flatten()
        
    return imag_part
        
        
