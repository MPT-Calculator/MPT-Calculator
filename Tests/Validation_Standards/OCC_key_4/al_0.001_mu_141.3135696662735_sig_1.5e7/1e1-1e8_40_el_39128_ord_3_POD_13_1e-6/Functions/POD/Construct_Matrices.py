from ngsolve import *
import scipy.sparse as sp
import numpy as np


def Construct_Matrices(mesh, fes,fes2,Theta0,u1Truncated,u2Truncated,u3Truncated,sigma,mu,inout, xivec, Integration_Order, Additional_Int_Order):
    """
    James Elgy - 2023
    Function to construct the

    Parameters
    ----------
    mesh NGSolve mesh
    fes Theta0 finite element space
    fes2 Theta1 finite element space
    Theta0 Nd0 x 3 real Theta0 solution vector
    u1Truncated Complex Nd x M left singular matrix for i = 0
    u2Truncated Complex Nd x M left singular matrix for i = 1
    u3Truncated Complex Nd x M left singular matrix for i = 2
    sigma Coefficient function for conductivity
    mu CoefficientFunction for relative permeability
    inout CoefficientFunction 1 inside 0 outside object
    xivec 1x3 𝐞ᵢ × ξ array
    Integration_Order Int order of integration for the non-finite element integrations
    Additional_Int_Order Int additional int order for bilinear and linear forms.

    Returns
    -------

    """



    # Real part:
    # Constructing K as real Nd x Nd matrix
    u1, v1 = fes2.TnT()
    K = BilinearForm(fes2, symmetric=True)
    K += SymbolicBFI(inout * mu ** (-1) * curl(u1) * Conj(curl(v1)), bonus_intorder=Additional_Int_Order)
    K += SymbolicBFI((1 - inout) * curl(u1) * Conj(curl(v1)), bonus_intorder=Additional_Int_Order)
    K.Assemble()

    # Converting to Scipy Sparse.
    rows, cols, vals = K.mat.COO()
    del K
    K = sp.csr_matrix((vals, (rows, cols)))
    del rows, cols, vals

    # For faster computation of tensor coefficients, we pre and post multiply with Ui to make real MxM.
    # Equivalent to Kᴹ from paper.
    KM_11 = np.conj(np.transpose(u1Truncated)) @ K @ u1Truncated
    KM_22 = np.conj(np.transpose(u2Truncated)) @ K @ u2Truncated
    KM_33 = np.conj(np.transpose(u3Truncated)) @ K @ u3Truncated
    KM_21 = np.conj(np.transpose(u2Truncated)) @ K @ u1Truncated
    KM_31 = np.conj(np.transpose(u3Truncated)) @ K @ u1Truncated
    KM_32 = np.conj(np.transpose(u3Truncated)) @ K @ u2Truncated

    del K
    KM_array = [KM_11, KM_22, KM_33, KM_21, KM_31, KM_32]



    # Imaginary Part:
    # Imaginary integral is split into 9 vector-matrix-vector products.
    u0, v0 = fes.TnT()

    # Building 𝐂 as real Nd x Nd matrix
    C = BilinearForm(fes2, symmetric=True)
    C += SymbolicBFI(sigma * inout * (u1 * v1), bonus_intorder=Additional_Int_Order)
    C.Assemble()

    # Convert to scipy sparse.
    rows, cols, vals = C.mat.COO()
    del C
    C = sp.csr_matrix((vals, (rows, cols)))
    del rows, cols, vals

    # Equivalent to 𝐂ᴹ from paper. as complex M times M matrix
    CM_11 = np.conj(np.transpose(u1Truncated)) @ C @ u1Truncated
    CM_22 = np.conj(np.transpose(u2Truncated)) @ C @ u2Truncated
    CM_33 = np.conj(np.transpose(u3Truncated)) @ C @ u3Truncated
    CM_21 = np.conj(np.transpose(u2Truncated)) @ C @ u1Truncated
    CM_31 = np.conj(np.transpose(u3Truncated)) @ C @ u1Truncated
    CM_32 = np.conj(np.transpose(u3Truncated)) @ C @ u2Truncated

    del C
    CM_array = [CM_11, CM_22, CM_33, CM_21, CM_31, CM_32]


    # 𝐂¹ matrix based on theta0 basis functions:
    # Building bold M as real Nd0 x Nd0 matrix
    C1 = BilinearForm(fes, symmetric=True)
    C1 += SymbolicBFI(sigma * inout * (u0 * v0), bonus_intorder=Additional_Int_Order)
    C1.Assemble()

    # Convert to scipy sparse.
    rows, cols, vals = C1.mat.COO()
    del C1
    C1 = sp.csr_matrix((vals, (rows, cols)))
    del rows, cols, vals

    # Building scalar (𝐨ⱼ)ᵀ 𝐂¹ 𝐨ᵢ
    ot_C1_o_11 = (np.transpose(Theta0[:, 0])) @ C1 @ Theta0[:, 0]
    ot_C1_o_22 = (np.transpose(Theta0[:, 1])) @ C1 @ Theta0[:, 1]
    ot_C1_o_33 = (np.transpose(Theta0[:, 2])) @ C1 @ Theta0[:, 2]
    ot_C1_o_21 = (np.transpose(Theta0[:, 1])) @ C1 @ Theta0[:, 0]
    ot_C1_o_31 = (np.transpose(Theta0[:, 2])) @ C1 @ Theta0[:, 1]
    ot_C1_o_32 = (np.transpose(Theta0[:, 2])) @ C1 @ Theta0[:, 2]

    ot_C1_o_array = [ot_C1_o_11, ot_C1_o_22, ot_C1_o_33, ot_C1_o_21, ot_C1_o_31, ot_C1_o_32]

    # 𝐂² matrix as real Nd0 x Nd
    C2 = BilinearForm(fes2, symmetric=True)
    C2 += SymbolicBFI(sigma * inout * (u0 * v1), bonus_intorder=Additional_Int_Order)
    C2.Assemble()

    # Convert to scipy sparse.
    rows, cols, vals = C2.mat.COO()
    del C2
    C2 = sp.csr_matrix((vals, (rows, cols)))
    del rows, cols, vals

    # Equivalent to (𝐂²)ᴹ from paper. as complex Nd times M matrix
    C2M_1 = C2 @ u1Truncated
    C2M_2 = C2 @ u2Truncated
    C2M_3 = C2 @ u3Truncated
    del C2

    # Precomputing 𝐨ᵀ (𝐂²)ᴹ
    ot_C2M_11 = np.transpose(Theta0[:, 0]) @ C2M_1
    ot_C2M_22 = np.transpose(Theta0[:, 1]) @ C2M_2
    ot_C2M_33 = np.transpose(Theta0[:, 2]) @ C2M_3
    ot_C2M_12 = np.transpose(Theta0[:, 0]) @ C2M_2
    ot_C2M_31 = np.transpose(Theta0[:, 2]) @ C2M_1
    ot_C2M_32 = np.transpose(Theta0[:, 2]) @ C2M_2

    ot_C2M_array = [ot_C2M_11, ot_C2M_22, ot_C2M_33, ot_C2M_12, ot_C2M_31, ot_C2M_32]

    # 𝐬ᵢ Array as real of size Nd0
    S = np.zeros((3, fes.ndof))
    for i in range(3):
        S_lf = LinearForm(fes2)
        S_lf += SymbolicLFI(sigma * inout * xivec[i] * v0, bonus_intorder=Additional_Int_Order)
        S_lf.Assemble()
        S[i, :] = S_lf.vec.FV().NumPy()[:]
        del S_lf

    # Computing (𝐬ᵢ)ᵀ 𝐨ⱼ
    Strans = S.transpose()
    del S
    St_o_11 = Theta0[:,0] @ Strans[:,0]
    St_o_22 = Theta0[:,1] @ Strans[:,1]
    St_o_33 = Theta0[:,2] @ Strans[:,2]
    St_o_12 = Theta0[:,1] @ Strans[:,0]
    St_o_31 = Theta0[:,2] @ Strans[:,0]
    St_o_32 = Theta0[:,2] @ Strans[:,2]

    St_o_array = [St_o_11, St_o_22, St_o_33, St_o_12, St_o_31, St_o_32]

    # 𝐭ᵢ Array as real of size Nd
    T = np.zeros((3, fes.ndof))
    for i in range(3):
        T_lf = LinearForm(fes2)
        T_lf += SymbolicLFI(sigma * inout * xivec[i] * v1, bonus_intorder=Additional_Int_Order)
        T_lf.Assemble()
        T[i, :] = T_lf.vec.FV().NumPy()[:]
        del T_lf

    # Computing (𝐭ᵢ)ᴹ
    TM_11 = T[0,:] @ u1Truncated
    TM_22 = T[1,:] @ u2Truncated
    TM_33 = T[2,:] @ u3Truncated
    TM_12 = T[1,:] @ u1Truncated
    TM_31 = T[2,:] @ u3Truncated
    TM_32 = T[2,:] @ u3Truncated

    TM_array = [TM_11, TM_22, TM_33, TM_12, TM_31, TM_32]
    del T

    # Computing c as real 3x3 matrix
    c = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            c[i, j] = Integrate(sigma * inout * xivec[i] * xivec[j], mesh, order=Integration_Order)



    return KM_array, CM_array, ot_C1_o_array, ot_C2M_array, St_o_array, TM_array, c


if __name__ == '__main__':
    help(Construct_Matrices)