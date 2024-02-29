# James Elgy - 02/06/2023

"""
Paul Ledger edit 28/02/2024 added drop_tol and symmetric_storage=True to reduce memory useage when creating large matrices
built using interior dofs
"""
import numpy as np
from matplotlib import pyplot as plt
from ngsolve import *
import scipy.sparse as sp
import gc

def Construct_Matrices(Integration_Order, Theta0Sol, bilinear_bonus_int_order, fes2, inout, mesh, mu_inv, sigma, sweepname,
                       u, u1Truncated, u2Truncated, u3Truncated, v, xivec, NumSolverThreads, drop_tol, ReducedSolve=True ):
    obtain_orders_iteratively = False
    tol_bilinear = 1e-10

    if NumSolverThreads != 'default':
        SetNumThreads(NumSolverThreads)
    if obtain_orders_iteratively is False:
        # Constructing 𝐊ᵢⱼ (eqn 7 from paper)
        # For the K bilinear forms, and also later bilinear and linear forms, we specify an integration order specific
        # to the postprocessing. See comment in main.py on the topic.
        u, v = fes2.TnT()
        K = BilinearForm(fes2, symmetric=True, delete_zero_elements =drop_tol,keep_internal=False, symmetric_storage=True)
        K += SymbolicBFI(inout * mu_inv * curl(u) * Conj(curl(v)), bonus_intorder=bilinear_bonus_int_order)
        K += SymbolicBFI((1 - inout) * curl(u) * Conj(curl(v)), bonus_intorder=bilinear_bonus_int_order)
        with TaskManager():
            K.Assemble()
        rows, cols, vals = K.mat.COO()
        del K
        Qsym = sp.csr_matrix((vals, (rows, cols)),shape=(fes2.ndof,fes2.ndof))
        del rows, cols, vals
        gc.collect()
    if obtain_orders_iteratively is True:
        u, v = fes2.TnT()
        rel_diff = 1
        counter = 1
        rel_diff_array = []
        ord_array = []
        bonus_intord = 0
        while (rel_diff > tol_bilinear) and (counter < 20):
            K = BilinearForm(fes2, symmetric=True, delete_zero_elements =drop_tol,keep_internal=False, symmetric_storage=True)
            K += SymbolicBFI(inout * mu_inv * curl(u) * Conj(curl(v)), bonus_intorder=bonus_intord)
            K += SymbolicBFI((1 - inout) * curl(u) * Conj(curl(v)), bonus_intorder=bonus_intord)
            with TaskManager():
                K.Assemble()

            rows, cols, vals = K.mat.COO()
            # del K
            if counter == 1:  # first iteration
                vals_old = vals
            else:
                vals_new = vals
                rel_diff = np.linalg.norm(vals_new - vals_old) / np.linalg.norm(vals_new)
                vals_old = vals
                rel_diff_array += [rel_diff]
                ord_array += [bonus_intord]

            # using bonus_intord =n and bonus_intord =n+1 gives the same result. So we use even orders.
            bonus_intord += 2
            counter += 1
            print(bonus_intord, rel_diff, np.linalg.norm(vals))

        plt.figure()
        plt.plot(ord_array, rel_diff_array, '*-', label='Relative Difference')
        plt.axhline(tol_bilinear, color='r', label='Tolerance')
        plt.xlabel('Integration Order')
        plt.ylabel('Relative Difference K')
        plt.yscale('log')
        plt.legend()
        plt.savefig('Results/' + sweepname + '/Graphs/BilinearForm_Convergence_K.pdf')

        # rows, cols, vals = K.mat.COO()
        del K
        Qsym = sp.csr_matrix((vals, (rows, cols)),shape=(fes2.ndof,fes2.ndof))
        del rows, cols, vals
    Q = Qsym + Qsym.T - sp.diags(Qsym.diagonal())
    del Qsym
    #rows,cols = Q.nonzero()
    #Q[cols,rows] = Q[rows,cols]
    #del rows,cols
        
    # For faster computation of tensor coefficients, we multiply with Ui before the loop.
    # This computes MxM 𝐊ᴹᵢⱼ. For each of the combinations ij we store the smaller matrix rather than recompute in
    # each case.
    if ReducedSolve == True:
        Q11 = np.conj(np.transpose(u1Truncated)) @ Q @ u1Truncated
        Q22 = np.conj(np.transpose(u2Truncated)) @ Q @ u2Truncated
        Q33 = np.conj(np.transpose(u3Truncated)) @ Q @ u3Truncated
        Q21 = np.conj(np.transpose(u2Truncated)) @ Q @ u1Truncated
        Q31 = np.conj(np.transpose(u3Truncated)) @ Q @ u1Truncated
        Q32 = np.conj(np.transpose(u3Truncated)) @ Q @ u2Truncated
    else:
        Q11 = Q22 = Q33 = Q21 = Q31 = Q32 = Q
    
    del Q
    Q_array = [Q11, Q22, Q33, Q21, Q31, Q32]
    # Similar for 𝐂ᴹᵢⱼ. refered to as A in code. For each of the combinations ij we store the smaller matrix rather
    # than recompute in each case.
    # Using the same basis functions for both the theta0 and theta1 problems allows us to reduce the number of
    # bilinear forms that need to be constructed.
    # For 𝐍ᴷ = (𝐍₀)ᴷ then 𝐂 = 𝐂¹ = 𝐂² and 𝐬ᵢ = 𝐭ᵢ. In this way we only need to consider 𝐂 (called A in code), 𝐬
    # (called E in code) and c (called G in code) from paper.
    if obtain_orders_iteratively is False:
        A = BilinearForm(fes2, symmetric=True, delete_zero_elements =drop_tol,keep_internal=False, symmetric_storage=True)
        A += SymbolicBFI(sigma * inout * (v * u), bonus_intorder=bilinear_bonus_int_order)
        with TaskManager():
            A.Assemble()
        rows, cols, vals = A.mat.COO()
        del A
        A_matsym = sp.csr_matrix((vals, (rows, cols)),shape=(fes2.ndof,fes2.ndof))
        del rows, cols, vals
        #print(np.shape(A_matsym))
        A_mat = A_matsym + A_matsym.T - sp.diags(A_matsym.diagonal())
        del A_matsym
        #rows,cols = A_mat.nonzero()
        #A_mat[cols,rows] = A_mat[rows,cols]
        #del rows,cols
        #print(np.shape(A_mat))
        gc.collect()
    else:
        rel_diff = 1
        counter = 1
        rel_diff_array = []
        ord_array = []
        bonus_intord = 0
        while (rel_diff > tol_bilinear) and (counter < 20):
            A = BilinearForm(fes2, symmetric=True, delete_zero_elements =drop_tol,keep_internal=False, symmetric_storage=True)
            A += SymbolicBFI(sigma * inout * (v * u), bonus_intorder=bonus_intord)
            with TaskManager():
                A.Assemble()
            rows, cols, vals = A.mat.COO()
            if counter == 1:  # first iteration
                vals_old = vals
            else:
                vals_new = vals
                rel_diff = np.linalg.norm(vals_new - vals_old) / np.linalg.norm(vals_new)
                vals_old = vals
                rel_diff_array += [rel_diff]
                ord_array += [bonus_intord]

            # using bonus_intord =n and bonus_intord =n+1 gives the same result. So we use even orders.
            bonus_intord += 2
            counter += 1
            print(bonus_intord, rel_diff)

        plt.figure()
        plt.plot(ord_array, rel_diff_array, '*-', label='Relative Difference')
        plt.axhline(tol_bilinear, color='r', label='Tolerance')
        plt.xlabel('Integration Order')
        plt.ylabel('Relative Difference $\mathbf{C}$')
        plt.yscale('log')
        plt.legend()
        plt.savefig('Results/' + sweepname + '/Graphs/BilinearForm_Convergence_C.pdf')

        del A
        A_matsym = sp.csr_matrix((vals, (rows, cols)))
        del rows, cols, vals
        A_mat = A_matsym + A_matsym.T - sp.diags(A_matsym.diagonal())
        #rows,cols = A_mat.nonzero()
        #A_mat[cols,rows] = A_mat[rows,cols]
        #del rows,cols
        gc.collect()
    E = np.zeros((3, fes2.ndof), dtype=complex)
    G = np.zeros((3, 3))
    for i in range(3):
        if obtain_orders_iteratively is False:
            E_lf = LinearForm(fes2)
            E_lf += SymbolicLFI(sigma * inout * xivec[i] * v, bonus_intorder=bilinear_bonus_int_order)
            E_lf.Assemble()
            E[i, :] = E_lf.vec.FV().NumPy()[:]
            del E_lf
        else:
            rel_diff = 1
            counter = 1
            rel_diff_array = []
            ord_array = []
            bonus_intord = 0
            while (rel_diff > tol_bilinear) and (counter < 20):
                E_lf = LinearForm(fes2)
                E_lf += SymbolicLFI(sigma * inout * xivec[i] * v, bonus_intorder=bonus_intord)
                E_lf.Assemble()
                vals = E_lf.vec.FV().NumPy()[:]
                if counter == 1:  # first iteration
                    vals_old = vals
                else:
                    vals_new = vals
                    rel_diff = np.linalg.norm(vals_new - vals_old) / np.linalg.norm(vals_new)
                    vals_old = vals
                    rel_diff_array += [rel_diff]
                    ord_array += [bonus_intord]

                # using bonus_intord =n and bonus_intord =n+1 gives the same result. So we use even orders.
                bonus_intord += 2
                counter += 1
                print(bonus_intord, rel_diff)

            E[i, :] = E_lf.vec.FV().NumPy()[:]
            del E_lf
            if i == 0:
                plt.figure()
            plt.plot(ord_array, rel_diff_array, '*-', label=f'Relative Difference, i={i + 1}')
            plt.axhline(tol_bilinear, color='r', label='Tolerance')
            plt.xlabel('Integration Order')
            plt.ylabel(r'Relative Difference $\mathbf{s}_{i}$')
            plt.yscale('log')
            plt.legend()
            plt.savefig('Results/' + sweepname + '/Graphs/LinearForm_Convergence_s.pdf')

        for j in range(3):
            G[i, j] = Integrate(sigma * inout * xivec[i] * xivec[j], mesh, order=Integration_Order)
    H = E.transpose()
    print(' Built K, Q, E, and G')


    # Testing:
    # run_test_comparison(u,v, sigma, xivec, inout, mesh, Theta0Sol, Lower_Sols, u1Truncated, fes, fes2)
    # Similarly for the imaginary part, we multiply with the theta0 sols beforehand.
    A_mat_t0_1 = (A_mat) @ Theta0Sol[:, 0]
    A_mat_t0_2 = (A_mat) @ Theta0Sol[:, 1]
    A_mat_t0_3 = (A_mat) @ Theta0Sol[:, 2]
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
    T_array = [T11, T22, T33, T21, T31, T32]
    # At this point, we have constructed each of the main matrices we need and obtained the reduced A matrix. The
    # larger bilinear form can therefore be removed to save memory.
    del A_mat
    At0_array = [A_mat_t0_1, A_mat_t0_2, A_mat_t0_3]
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
        
    UAt0_conj = [UAt011_conj, UAt022_conj, UAt033_conj, UAt012_conj, UAt013_conj, UAt023_conj]
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
    
    UAt0U_array = [UAt011, UAt022, UAt033, UAt021, UAt031, UAt032]
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
    c1_array = [c1_11, c1_22, c1_33, c1_21, c1_31, c1_32]
    c5_array = [c5_11, c5_22, c5_33, c5_21, c5_31, c5_32]
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
    c8_array = [c8_11, c8_22, c8_33, c8_21, c8_31, c8_32]
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
    EU_array_conj = [EU_11, EU_22, EU_33, EU_21, EU_31, EU_32]
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
        
    UH_array = [UH_11, UH_22, UH_33, UH_21, UH_31, UH_32]
    return At0_array, EU_array_conj, Q_array, T_array, UAt0U_array, UAt0_conj, UH_array, c1_array, c5_array, c7, c8_array
