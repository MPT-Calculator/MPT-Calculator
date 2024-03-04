# James Elgy - 26/04/2023

import numpy as np
from matplotlib import pyplot as plt
from ngsolve import *
from scipy import sparse as sp

def run_test_comparison(u,v, sigma, xivec, inout, mesh, Theta0Sol, Sols, u1Truncated, fes, fes2):

    Theta_0i = GridFunction(fes)
    Theta_1i = GridFunction(fes2)

    bonus_int_order = 10
    int_order = 12

    alpha = 0.01
    Mu0=4*np.pi*10**(-7)
    nu_no_omega = Mu0 * (alpha ** 2)

    u, v = fes.TnT()

    # Wi = Sols[:, omega_index, 0]
    g1 = np.squeeze(np.asarray(Sols))[:,-1, 0]
    W1 = np.dot(u1Truncated, g1).flatten()
    omega = 1e8
    Theta_0i.vec.FV().NumPy()[:] = Theta0Sol[:, 0]
    Theta_1i.vec.FV().NumPy()[:] = W1
    xii = xivec[0]
    nu_no_sigma = Mu0 * alpha**2 * omega

    # Setting Matrices:

    A = BilinearForm(fes, symmetric=True)
    A += SymbolicBFI(sigma * inout * (v * u), bonus_intorder=bonus_int_order)
    A.Assemble()
    rows, cols, vals = A.mat.COO()
    del A
    A = sp.csr_matrix((vals, (rows, cols)))
    del rows, cols, vals

    E = np.zeros((fes.ndof, 1), dtype=complex)
    E_lf = LinearForm(fes)
    E_lf += SymbolicLFI(sigma * inout * xii * v, bonus_intorder=bonus_int_order)
    E_lf.Assemble()
    E[:, 0] = E_lf.vec.FV().NumPy()[:]
    del E_lf

    E_trans_reduced = np.transpose(E) @ u1Truncated
    E_trans_reduced_conj = np.transpose(E) @ np.conj(u1Truncated)

    G = Integrate(sigma * inout * xivec[0] * xivec[0], mesh, order=int_order)

    u, v = fes2.TnT()
    C = BilinearForm(fes2, symmetric=True)
    C += SymbolicBFI(sigma * inout * u * v, bonus_intorder=bonus_int_order)
    C.Assemble()
    rows, cols, vals = C.mat.COO()
    del C
    C = sp.csr_matrix((vals, (rows, cols)))
    del rows, cols, vals

    C_reduced = np.conj(np.transpose(u1Truncated)) @ C @ u1Truncated

    C2 = BilinearForm(fes2, symmetric=True)
    C2 += SymbolicBFI(sigma * inout * u * v, bonus_intorder=bonus_int_order)
    C2.Assemble()
    rows, cols, vals = C2.mat.COO()
    del C2
    C2 = sp.csr_matrix((vals, (rows, cols)))
    del rows, cols, vals

    C2_reduced = C2 @ u1Truncated
    C2_reduced_conj = C2 @ np.conj(u1Truncated)


    # Term 1:
    print('Term 1:  𝐨ᵢᵀ 𝐂 𝐨ᵢ')
    int_term1 = np.real(Integrate(inout * nu_no_sigma * sigma * Theta_0i * Theta_0i, mesh, order=int_order))
    mat_term1 = nu_no_sigma * np.real(np.transpose(Theta0Sol[:, 0]) @ A @ Theta0Sol[:, 0])
    print(f'int: {int_term1}')
    print(f'mat: {mat_term1}')

    print('')
    #Term 2:
    print('Term 2: cᵢᵢ')
    int_term2 = np.real(Integrate(inout * nu_no_sigma * sigma * xii * xii, mesh, order=int_order))
    mat_term2 =  nu_no_sigma * G
    print(f'int: {int_term2}')
    print(f'mat: {mat_term2}')

    print('')
    #Term 3:
    print('Term 3: 𝐬ᵢᵀ 𝐨ᵢ')
    int_term3 = np.real(Integrate(inout * nu_no_sigma * sigma * Theta_0i * xii, mesh, order=int_order))
    mat_term3 =  nu_no_sigma * np.real(np.transpose(E) @ Theta0Sol[:,0])
    print(f'int: {int_term3}')
    print(f'mat: {mat_term3}')

    print('')
    # Term 4:
    print('Term 4: 𝐬ᵢᵀ 𝐨ᵢ')
    int_term4 = np.real(Integrate(inout * nu_no_sigma * sigma * Theta_0i * xii, mesh, order=int_order))
    mat_term4 = nu_no_sigma * np.real(np.transpose(E) @ Theta0Sol[:, 0])
    print(f'int: {int_term4}')
    print(f'mat: {mat_term4}')

    print('')
    # Term 5:
    print('Term 5: ̅𝐪ᵀᵢ 𝐂 𝐪ᵢ')
    int_term5 = np.real(Integrate(inout * nu_no_sigma * sigma * Theta_1i * Conj(Theta_1i), mesh, order=int_order))
    mat_term5_full = nu_no_sigma * np.real(np.transpose(np.conj(W1)) @ C @ W1)
    mat_term5 = nu_no_sigma * np.real(np.transpose(np.conj(g1)) @ C_reduced @ g1)
    print(f'int: {int_term5}')
    print(f'mat reduced: {mat_term5}')
    print(f'mat full: {mat_term5_full}')

    print('')
    # Term 6:
    print('Term 6: 𝐨ᵀᵢ 𝐂⁽²⁾  𝐪ᵢ')
    int_term6 = np.real(Integrate(inout * nu_no_sigma * sigma * Theta_0i * Theta_1i, mesh, order=int_order))
    mat_term6_full = nu_no_sigma * np.real(Theta0Sol[:,0] @ C2 @ W1)
    mat_term6 = nu_no_sigma * np.real(Theta0Sol[:,0] @ C2_reduced @ g1)
    print(f'int: {int_term6}')
    print(f'mat reduced: {mat_term6}')
    print(f'mat full: {mat_term6_full}')

    print('')
    # Term 7:
    print('Term 7: 𝐨ᵀᵢ 𝐂⁽²⁾  ̅𝐪ᵢ')
    int_term7 = np.real(Integrate(inout * nu_no_sigma * sigma * Theta_0i * Conj(Theta_1i), mesh, order=int_order))
    mat_term7_full = nu_no_sigma * np.real(np.transpose(Theta0Sol[:,0]) @ C2 @ np.conj(W1))
    mat_term7 = nu_no_sigma * np.real(np.transpose(Theta0Sol[:,0]) @ C2_reduced_conj @ np.conj(g1))
    print(f'int: {int_term7}')
    print(f'mat reduced: {mat_term7}')
    print(f'mat full: {mat_term7_full}')

    print('')
    # Term 8:
    print('Term 8: 𝐭ᵀᵢ 𝐪ᵢ')
    int_term8 = np.real(Integrate(inout * nu_no_sigma * sigma * Theta_1i * xii, mesh, order=int_order))
    mat_term8_full = nu_no_sigma * np.real(np.transpose(E) @ W1)
    mat_term8 = nu_no_sigma * np.real(E_trans_reduced @ g1)
    print(f'int: {int_term8}')
    print(f'mat reduced: {mat_term8}')
    print(f'mat full: {mat_term8_full}')

    print('')
    # Term 9:
    print('Term 9: 𝐭ᵀᵢ ̅𝐪ᵢ')
    int_term9 = np.real(Integrate(inout * nu_no_sigma * sigma * Theta_1i * xii, mesh, order=int_order))
    mat_term9_full = nu_no_sigma * np.real(np.transpose(E) @ np.conj(W1))
    mat_term9 = nu_no_sigma * np.real(E_trans_reduced_conj @ np.conj(g1))
    print(f'int: {int_term9}')
    print(f'mat reduced: {mat_term9}')
    print(f'mat full: {mat_term9_full}')

    sum_int = int_term1 + int_term2 + int_term3 + int_term4 + int_term5 + int_term6 + int_term7 + int_term8 + int_term9
    sum_mat = mat_term1 + mat_term2 + mat_term3 + mat_term4 + mat_term5 + mat_term6 + mat_term7 + mat_term8 + mat_term9
    sum_mat_full = mat_term1 + mat_term2 + mat_term3 + mat_term4 + mat_term5_full + mat_term6_full + mat_term7_full + mat_term8_full + mat_term9_full

    print('')
    print('sum')
    print(f'Int: {sum_int}')
    print(f'mat reduced: {sum_mat}')
    print(f'mat full: {sum_mat_full}')

    int_meth = (alpha**3 / 4) * sum_int
    mat_meth = (alpha**3 / 4) * sum_mat

    full_int = (alpha**3 / 4) * Integrate(
                        inout * nu_no_sigma * sigma * ((Theta_1i + Theta_0i + xii) * (Conj(Theta_1i) + Theta_0i + xii)), mesh,
                        order=int_order).real

    print('')
    print(f'Int Method (full integration): {full_int}')
    print(f'Int Method (sum of parts): {int_meth}')
    print(f'Mat Method: {mat_meth}')

    print('end')




