# James Elgy - 02/06/2023

import numpy as np
from matplotlib import pyplot as plt
from ngsolve import *
# from netgen.meshing import *
import warnings
import os
import gc

def BilinearForms_Check(mesh, order, mu_inv, sigma, inout, bilinearform_tol, max_iter, curve_order, starting_order, sweepname, NumSolverThreads, drop_tol):
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
    drop_tol - float - Tolerance for dropping near 0 values in assembled matrices including interior

    Returns
    -------
    bonus_intord - int - converged order of integration
    """

    """
    Paul Ledger 2024 edit
    Avoid passing large amounts of data between an NG-Solve matrix and a Numpy/SciPy sparse array
    We only need the Forbenious norm and by viewing the matrix AsVector this can be computed

    Also update critera to save resources. Rather than checking the norm of the difference of values
    Instead just check the absolute difference of the norms - we only use this for estimating the
    order of integration for computing the MPTs after all

    Use delete_zero_elements =drop_tol to further reduce memory of large K and A containing interiors
    dofs that are only used when post-processing to compute the MPT and during the POD
    """

    print('Running Bilinear Forms Check')
    
    if NumSolverThreads != 'default':
        SetNumThreads(NumSolverThreads)
    
    
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


    while (rel_diff > bilinearform_tol) and (counter < max_iter):
        print(f'K: Iteration {counter}: bonus_intord = {bonus_intord}')
        K = BilinearForm(fes2, symmetric=True, delete_zero_elements =drop_tol, keep_internal=False, symmetric_storage=True)
        K += SymbolicBFI(inout * mu_inv * curl(u) * (curl(v)), bonus_intorder=bonus_intord)
        K += SymbolicBFI((1 - inout) * curl(u) * (curl(v)), bonus_intorder=bonus_intord)
        #K += SymbolicBFI(epsi * u * v, bonus_intorder=bonus_intord)
        with TaskManager():
            K.Assemble()
        #print(K.mat.nze)
        
        if counter == 1:  # first iteration
            nvalsold = np.linalg.norm(K.mat.AsVector()[:])
        else:
            last_rel_diff = rel_diff
            nvals = np.linalg.norm(K.mat.AsVector()[:])
            rel_diff = np.abs(nvals-nvalsold)/nvals
            nvalsold = nvals
            #vals_old = vals
            rel_diff_array += [rel_diff]
            ord_array += [bonus_intord]
        del K
        if counter > 1 and rel_diff >= last_rel_diff:
            # No convergence
            print("No convergence - exit loop and switch to linear geometry")
            counter=max_iter
            break

        #vals = np.zeros(nz)  # reset vals

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
    if counter >= max_iter:
        warnings.warn("K Bilinear Form did not converge. Trying again with linear geometry.")
        if curve_order > 1:
            gc.collect()
            return BilinearForms_Check(mesh, order, mu_inv, sigma, inout, bilinearform_tol, max_iter, 1, starting_order, sweepname, NumSolverThreads, drop_tol)
            curve_order = 1
            mesh.Curve(1)
        else:
            warnings.warn("K Bilinear Form did not converge with linear geometry. This may indicate a mesh error.")
    
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



    while (rel_diff > bilinearform_tol) and (counter < max_iter):
        print(f'C: Iteration {counter}: bonus_intord = {bonus_intord}')
        A = BilinearForm(fes2, symmetric=True, delete_zero_elements =drop_tol, keep_internal=False, symmetric_storage=True)
        A += SymbolicBFI(sigma * inout * (v * u), bonus_intorder=bonus_intord)
        with TaskManager():
            A.Assemble()
        #print(A.mat.nze)


        #rows[:], cols[:], vals[:] = A.mat.COO()
        #del A
        #if counter == 1:  # first iteration
        #    vals_old = vals
        #else:
        #    last_rel_diff = rel_diff
        #    rel_diff = np.linalg.norm(vals - vals_old) / np.linalg.norm(vals)
        #    vals_old = vals
        #    rel_diff_array += [rel_diff]
        #    ord_array += [bonus_intord]

        if counter == 1:  # first iteration
            nvalsold = np.linalg.norm(A.mat.AsVector()[:])
        else:
            last_rel_diff = rel_diff
            nvals = np.linalg.norm(A.mat.AsVector()[:])
            rel_diff = np.abs(nvals-nvalsold)/nvals
            nvalsold = nvals
            #vals_old = vals
            rel_diff_array += [rel_diff]
            ord_array += [bonus_intord]
        del A

        if counter > 1 and rel_diff >= last_rel_diff:
            # No convergence
            print("No convergence - exit loop and switch to linear geometry")
            counter=max_iter
            break

        #vals = np.zeros(nz)  # reset vals

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
    if counter >= max_iter:
        warnings.warn("C Bilinear Form did not converge. Trying again with linear geometry.")
        if curve_order > 1:
            gc.collect()
            return BilinearForms_Check(mesh, order, mu_inv, sigma, inout, bilinearform_tol, max_iter, 1, starting_order, sweepname, NumSolverThreads, drop_tol)
            curve_order = 1
            mesh.Curve(1)
        else:
            warnings.warn("C Bilinear Form did not converge with linear geometry. This may indicate a mesh error.")
    C_order = bonus_intord
    #
    print(f'C Bilinear Form Converged using order {C_order}')

    gc.collect()
    return max([K_order, C_order])

