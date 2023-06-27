# James Elgy - 02/06/2023

import numpy as np
from matplotlib import pyplot as plt
from ngsolve import *
# from netgen.meshing import *
import warnings
import os

def BilinearForms_Check(mesh, order, mu_inv, sigma, inout, bilinearform_tol, max_iter, curve_order, starting_order, sweepname):
    """
    James Elgy - 2023
    Function to compute and check the convergence of the postprocessing bilinear forms.
    Parameters
    ----------
    mesh - NGmesh - Object mesh
    order - int - Order of the Bilinear Form
    mu - CoefficientFunction - relative permeability
    sigma - CoefficientFunction - Conductivity
    inout - CoefficientFunction - 1 inside 0 outside.
    bilinearform_tol - float - Tolerance for convergence
    max_iter - int - Maximum number of iterations
    curve_order - int - order of the curved geometry

    Returns
    -------
    bonus_intord - int - converged order of integration
    """

    print('Running Bilinear Forms Check')
    
    
    # Making directory to save graphs:
    try:
        os.mkdir('Results/' + sweepname + '/Graphs/')
    except:
        pass

    # Setting mesh curve order so that we can recheck for linear geometry:
    mesh.Curve(curve_order)
    print('Mesh curve order set to: ', curve_order)

    # Checking K:
    # defining finite element space.
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes2 = HCurl(mesh, order=order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)


    # Creating the bilinear form
    u, v = fes2.TnT()
    rel_diff = 1
    counter = 1
    rel_diff_array = []
    ord_array = []
    bonus_intord = starting_order

    # Sparsity pattern won't change, so we can create K using order 0 and obtain the number of non-zero entries
    # for preallocation
    K = BilinearForm(fes2, symmetric=True)
    K += SymbolicBFI(inout * mu_inv * curl(u) * (curl(v)), bonus_intorder=bonus_intord)
    K += SymbolicBFI((1 - inout) * curl(u) * (curl(v)), bonus_intorder=bonus_intord)
    K.Assemble()

    _, _, s = K.mat.COO()
    rows = np.zeros(len(s))
    cols = np.zeros(len(s))
    vals = np.zeros(len(s))

    while (rel_diff > bilinearform_tol) and (counter < max_iter):
        print(f'K: Iteration {counter}: bonus_intord = {bonus_intord}')
        K = BilinearForm(fes2, symmetric=True)
        K += SymbolicBFI(inout * mu_inv * curl(u) * (curl(v)), bonus_intorder=bonus_intord)
        K += SymbolicBFI((1 - inout) * curl(u) * (curl(v)), bonus_intorder=bonus_intord)
        K.Assemble()

        # K is real here, so we discard imaginary part.
        rows[:], cols[:], vals[:] = K.mat.COO()
        # del K
        if counter == 1:  # first iteration
            vals_old = vals
        else:
            vals_new = vals
            rel_diff = np.linalg.norm(vals_new - vals_old) / np.linalg.norm(vals_new)
            vals_old = vals
            rel_diff_array += [rel_diff]
            ord_array += [bonus_intord]

        vals = np.zeros(len(s))  # reset vals

        # using bonus_intord =n and bonus_intord =n+1 gives the same result (?). So we use even orders.
        bonus_intord += 2
        counter += 1

    plt.figure()
    plt.title(f'Curve order: {curve_order}')
    plt.plot(ord_array, rel_diff_array, '*-', label='Relative Difference K')
    plt.axhline(bilinearform_tol, color='r', label='Tolerance')
    plt.xlabel('Integration Order')
    plt.ylabel('Relative Difference K')
    plt.yscale('log')
    plt.legend()
    plt.savefig('Results/' + sweepname + '/Graphs/K_Convergence.pdf')



    # Checking for convergence:
    #if counter >= max_iter:
    #    warnings.warn("K Bilinear Form did not converge. Trying again with linear geometry.")
    #    if curve_order > 1:
    #        return BilinearForms_Check(mesh, order, mu_inv, sigma, inout, bilinearform_tol, max_iter, 1, starting_order, sweepname)
    #        curve_order = 1
    #        mesh.Curve(1)
    #    else:
    #        warnings.warn("K Bilinear Form did not converge with linear geometry. This may indicate a mesh error.")
    #
    K_order = bonus_intord
    print(f'K Bilinear Form Converged using order {K_order}')

    # Solving for C
    # Creating the bilinear form
    u, v = fes2.TnT()
    rel_diff = 1
    counter = 1
    rel_diff_array = []
    ord_array = []
    bonus_intord = starting_order

    # Sparsity pattern won't change so we can create A using order 0 and obtain the number of non-zero entries
    # for preallocation
    A = BilinearForm(fes2, symmetric=True)
    A += SymbolicBFI(sigma * inout * (v * u), bonus_intorder=bonus_intord)
    A.Assemble()

    _, _, s = A.mat.COO()
    rows = np.zeros(len(s))
    cols = np.zeros(len(s))
    vals = np.zeros(len(s))

    while (rel_diff > bilinearform_tol) and (counter < max_iter):
        print(f'C: Iteration {counter}: bonus_intord = {bonus_intord}')
        A = BilinearForm(fes2, symmetric=True)
        A += SymbolicBFI(sigma * inout * (v * u), bonus_intorder=bonus_intord)
        A.Assemble()

        rows[:], cols[:], vals[:] = A.mat.COO()
        # del K
        if counter == 1:  # first iteration
            vals_old = vals
        else:
            vals_new = vals
            rel_diff = np.linalg.norm(vals_new - vals_old) / np.linalg.norm(vals_new)
            vals_old = vals
            rel_diff_array += [rel_diff]
            ord_array += [bonus_intord]

        vals = np.zeros(len(s))  # reset vals

        # using bonus_intord =n and bonus_intord =n+1 gives the same result (?). So we use even orders.
        bonus_intord += 2
        counter += 1

    plt.figure()
    plt.title(f'Curve order: {curve_order}')
    plt.plot(ord_array, rel_diff_array, '*-', label='Relative Difference C')
    plt.axhline(bilinearform_tol, color='r', label='Tolerance')
    plt.xlabel('Integration Order')
    plt.ylabel('Relative Difference C')
    plt.yscale('log')
    plt.legend()
    plt.savefig('Results/' + sweepname + '/Graphs/C_Convergence.pdf')



    # Checking for convergence:
    #if counter >= max_iter:
    #    warnings.warn("C Bilinear Form did not converge. Trying again with linear geometry.")
    #    if curve_order > 1:
    #        return BilinearForms_Check(mesh, order, mu_inv, sigma, inout, bilinearform_tol, max_iter, 1, starting_order, sweepname)
    #        curve_order = 1
    #        mesh.Curve(1)
    #    else:
    #        warnings.warn("C Bilinear Form did not converge with linear geometry. This may indicate a mesh error.")
    C_order = bonus_intord
    #
    print(f'C Bilinear Form Converged using order {C_order}')

    return max([K_order, C_order])

