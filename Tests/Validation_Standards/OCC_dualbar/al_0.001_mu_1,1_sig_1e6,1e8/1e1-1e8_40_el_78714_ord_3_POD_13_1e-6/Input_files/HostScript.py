import matplotlib.pyplot as plt
import time
from main import *
from Functions.Helper_Functions.exact_sphere import exact_sphere
import os
import itertools

def test_OCC():
    ReturnDict_geo = main(geometry='Cut_sphere_cube.geo', use_POD=True)
    ReturnDict_occ = main(geometry='OCC_cut_sphere_cube.py', use_OCC=True, use_POD=True)
    ReturnDict_step = main(geometry='OCC_cut_sphere_cube_step.py', use_OCC=True, use_POD=True)

    plt.figure()
    for i in range(9):
        if i == 0:
            plt.semilogx(ReturnDict_geo['FrequencyArray'], ReturnDict_geo['TensorArray'][:, i].real, 'r', label='.geo')
            plt.semilogx(ReturnDict_occ['FrequencyArray'], ReturnDict_occ['TensorArray'][:, i].real, 'g', label='OCC')
            plt.semilogx(ReturnDict_step['FrequencyArray'], ReturnDict_step['TensorArray'][:, i].real, 'b',
                         label='.step')
        else:
            plt.semilogx(ReturnDict_geo['FrequencyArray'], ReturnDict_geo['TensorArray'][:, i].real, 'r')
            plt.semilogx(ReturnDict_occ['FrequencyArray'], ReturnDict_occ['TensorArray'][:, i].real, 'g')
            plt.semilogx(ReturnDict_step['FrequencyArray'], ReturnDict_step['TensorArray'][:, i].real, 'b')

    plt.xlabel('$\omega$ [rad/s]')
    plt.ylabel('Real Tensor Coefficients [m]')
    plt.title('Tensor Coefficients Comparison for .geo, OCC, and .step formats')
    plt.legend()
    plt.figure()
    for i in range(9):
        if i == 0:
            plt.semilogx(ReturnDict_geo['FrequencyArray'], ReturnDict_geo['TensorArray'][:, i].imag, 'r', label='.geo')
            plt.semilogx(ReturnDict_occ['FrequencyArray'], ReturnDict_occ['TensorArray'][:, i].imag, 'g', label='OCC')
            plt.semilogx(ReturnDict_step['FrequencyArray'], ReturnDict_step['TensorArray'][:, i].imag, 'b',
                         label='.step')
        else:
            plt.semilogx(ReturnDict_geo['FrequencyArray'], ReturnDict_geo['TensorArray'][:, i].imag, 'r')
            plt.semilogx(ReturnDict_occ['FrequencyArray'], ReturnDict_occ['TensorArray'][:, i].imag, 'g')
            plt.semilogx(ReturnDict_step['FrequencyArray'], ReturnDict_step['TensorArray'][:, i].imag, 'b')

    plt.xlabel('$\omega$ [rad/s]')
    plt.ylabel('Imag Tensor Coefficients [m]')
    plt.title('Tensor Coefficients Comparison for .geo, OCC, and .step formats')
    plt.legend()


def test_key_4_against_paper():
    ReturnDict = main(geometry='remeshed_matt_key_4_steel.geo', use_POD=True, order=3)
    paper_eigenvalues = np.genfromtxt(
        r'Testing/key_4_paper_results/al_0.001_mu_141.3135696662735_sig_1.5e7/1e1-1e8_40_el_40229_ord_3_POD_13_1e-6/Data/Eigenvalues.csv',
        dtype=complex, delimiter=', ')
    paper_frequencies = np.genfromtxt(
        r'Testing/key_4_paper_results/al_0.001_mu_141.3135696662735_sig_1.5e7/1e1-1e8_40_el_40229_ord_3_POD_13_1e-6/Data/Frequencies.csv')

    plt.figure()
    for i in range(3):
        if i == 0:
            plt.semilogx(paper_frequencies, paper_eigenvalues[:, i].real, 'r', label='paper eigenvalues')
            plt.semilogx(ReturnDict['FrequencyArray'], ReturnDict['EigenValues'][:, i].real, 'b',
                         label='MPT eigenvalues')
        else:
            plt.semilogx(paper_frequencies, paper_eigenvalues[:, i].real, 'r')
            plt.semilogx(ReturnDict['FrequencyArray'], ReturnDict['EigenValues'][:, i].real, 'b')

    plt.xlabel('$\omega$ [rad/s]')
    plt.ylabel('Real Eigenvalues [m]')
    plt.title('Eigenvalues Comparison for key 4')
    plt.legend()

    plt.figure()
    for i in range(3):
        if i == 0:
            plt.semilogx(paper_frequencies, paper_eigenvalues[:, i].imag, 'r', label='paper eigenvalues')
            plt.semilogx(ReturnDict['FrequencyArray'], ReturnDict['EigenValues'][:, i].imag, 'b',
                         label='MPT eigenvalues')
        else:
            plt.semilogx(paper_frequencies, paper_eigenvalues[:, i].imag, 'r')
            plt.semilogx(ReturnDict['FrequencyArray'], ReturnDict['EigenValues'][:, i].imag, 'b')

    plt.xlabel('$\omega$ [rad/s]')
    plt.ylabel('Imag Eigenvalues [m]')
    plt.title('Eigenvalues Comparison for key 4')
    plt.legend()
    plt.savefig('Test_key4_comparison.pdf', format='pdf')


def hp_refinement_sphere():
    # P Refinement
    comparison_eig_p = np.zeros((40, 6), dtype=complex)
    comparison_ndofs_p = np.zeros(6)
    start_time = time.time()
    geo = 'OCC_sphere_prism_32.geo'
    for p in [0, 1, 2, 3, 4, 5]:
        print('Solving for order =', p)
        Return_Dict = main(use_POD=True, order=p,geometry=geo)
        comparison_eig_p[:, p] = Return_Dict['EigenValues'][:, 0]
        comparison_ndofs_p[p] = Return_Dict['NDOF'][1]
    stop_time = time.time()

    plt.figure()
    for p in [0, 1, 2, 3,4,5]:
        plt.semilogx(Return_Dict['FrequencyArray'], comparison_eig_p[:, p].real, label=f'order {p}')
    plt.legend()
    plt.xlabel('$\omega$, [rad/s]')
    plt.ylabel('$\lambda_1(\mathcal{R} + \mathcal{N}^0)$')

    plt.figure()
    for p in [0, 1, 2, 3,4,5]:
        plt.semilogx(Return_Dict['FrequencyArray'], comparison_eig_p[:, p].imag, label=f'order {p}')
    plt.legend()
    plt.xlabel('$\omega$, [rad/s]')
    plt.ylabel('$\lambda_1(\mathcal{I})$')

    # hp comparison to exact solution
    frequency_index = 39
    alpha = 1e-3
    sigma = 1e6
    mur = 32
    epsilon = 0
    omega = Return_Dict['FrequencyArray'][frequency_index]
    exact_solution = exact_sphere(alpha, epsilon, mur, sigma, omega)

    relative_error_real_p = np.zeros(6)
    relative_error_imag_p = np.zeros(6)


    for p in [0, 1, 2, 3,4,5]:
        relative_error_real_p[p] = np.abs((comparison_eig_p[frequency_index, p].real - exact_solution.real)) / np.abs(
            exact_solution.real)
        relative_error_imag_p[p] = np.abs((comparison_eig_p[frequency_index, p].imag - exact_solution.imag)) / np.abs(
            exact_solution.imag)



    plt.figure()
    plt.scatter(comparison_ndofs_p, relative_error_real_p, label='$p$ refinement')

    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.legend(loc='lower left')
    plt.ylabel('Relative Error $\lambda_1(\mathcal{R} + \mathcal{N}^0)$')
    plt.xlabel('NDOF')

    plt.figure()
    plt.scatter(comparison_ndofs_p, relative_error_imag_p, label='$p$ refinement')

    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.legend(loc='lower left')
    plt.ylabel('Relative Error $\lambda_1(\mathcal{I})$')
    plt.xlabel('NDOF')


def test_POD():
    geometry='Tetra.geo'
    sim_results = main(geometry=geometry, use_POD=True)
    sim_results_full = main(geometry=geometry, use_POD=False)

    plt.figure()
    for i in range(9):
        if i == 0:
            plt.semilogx(sim_results['FrequencyArray'], sim_results['TensorArray'][:,i].real, 'b', label='POD Sweep')
            plt.semilogx(sim_results_full['FrequencyArray'], sim_results_full['TensorArray'][:,i].real, 'r', label='Full Sweep')
        else:
            plt.semilogx(sim_results['FrequencyArray'], sim_results['TensorArray'][:, i].real, 'b')
            plt.semilogx(sim_results_full['FrequencyArray'], sim_results_full['TensorArray'][:, i].real, 'r')

    plt.legend()
    plt.xlabel('$\omega$, rad/s')
    plt.ylabel(r'$\tilde{\mathcal{R}}$, m$^3$')
    plt.figure()
    if i == 0:
        plt.semilogx(sim_results['FrequencyArray'], sim_results['TensorArray'][:, i].imag, 'b', label='POD Sweep')
        plt.semilogx(sim_results_full['FrequencyArray'], sim_results_full['TensorArray'][:, i].imag, 'r',
                     label='Full Sweep')
    else:
        plt.semilogx(sim_results['FrequencyArray'], sim_results['TensorArray'][:, i].imag, 'b')
        plt.semilogx(sim_results_full['FrequencyArray'], sim_results_full['TensorArray'][:, i].imag, 'r')
    plt.legend()
    plt.xlabel('$\omega$, rad/s')
    plt.ylabel(r'$\mathcal{I}$, m$^3$')

def full_suite_of_test_objects(recoverymode=True):
    """
    This function runs through each of the files listed in file list and returns the output for each.
    The idea is that this becomes a standard test suite of objects, against which we can test changes.
    Returns
    list of dictionaries containing output results for each simulation.
    -------

    """


    filelist = [
               # 'OCC_key_4.py',
                'OCC_thin_disc_magnetic_32.py',
                'OCC_sphere_prism_32.py',
				'OCC_dualbar.py',
                'OCC_bomblet_clamp_realistic_materials_hollow.py'
                ]
				
    alpha_list = [1e-3, 1e-2, 1e-2, 1e-2]
    order_list = [ 3, 3, 3, 4]


    plt.close('all')
    out = []
    for file, p, al in zip(filelist, order_list, alpha_list):
        print(f'Solving for file: {file}')
        if file == 'OCC_prism_copper_steel_1pcoin.py':
            cpus = 1
        else:
            cpus = 2
        Return_Dict = main(geometry=file, use_POD=True, use_OCC=True, order=p, cpus=cpus, alpha=al)
        out += [Return_Dict]
        #np.save('2302025_object_suite_old_ngsolve.npy', out)
    return out

def mat_int_method_comparison(object):
    toggle_integral(True)
    return_int = main(geometry=object, order=4, use_POD=True, use_OCC=True, start_stop=(1,8,160))

    toggle_integral(False)
    return_mat = main(geometry=object, order=4, use_POD=True, use_OCC=True, start_stop=(1, 8, 160))

    cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:pink']


    # Plotting Real component:
    plt.figure(1)
    count = 0
    for i in range(3):
        for j in range(i + 1):
            d = return_mat['TensorArray'].real.reshape(len(return_mat['FrequencyArray']), 3, 3)
            plt.semilogx(return_mat['FrequencyArray'],d[:, i, j],
                         label=f'matrix: i={i + 1},j={j + 1}', color=cols[count])
            count += 1

    count = 0
    for i in range(3):
        for j in range(i + 1):
            d = return_int['TensorArray'].real.reshape(len(return_int['FrequencyArray']), 3, 3)
            plt.semilogx(return_int['FrequencyArray'], d[:, i, j],
                         label=f'int: i={i + 1},j={j + 1}', color=cols[count], linestyle='--')
            count += 1

    plt.legend(prop={'size': 8}, loc=1)
    plt.xlabel('$\omega$, [rad/s]')
    plt.ylabel(r'$(\tilde{\mathcal{R}})_{ij}$, [m$^3$]')
    plt.ticklabel_format(style='sci', scilimits=(-4,4), axis='y')

    # Plotting Imag component:
    plt.figure(2)
    count = 0
    for i in range(3):
        for j in range(i + 1):
            d = return_mat['TensorArray'].imag.reshape(len(return_mat['FrequencyArray']), 3, 3)
            plt.semilogx(return_mat['FrequencyArray'],d[:, i, j],
                         label=f'matrix: i={i + 1},j={j + 1}', color=cols[count])
            count += 1

    count = 0
    for i in range(3):
        for j in range(i + 1):
            d = return_int['TensorArray'].imag.reshape(len(return_int['FrequencyArray']), 3, 3)
            plt.semilogx(return_int['FrequencyArray'], d[:, i, j],
                         label=f'int: i={i + 1},j={j + 1}', color=cols[count], linestyle='--')
            count += 1

    plt.legend(prop={'size': 8}, loc=1)
    plt.xlabel('$\omega$, [rad/s]')
    plt.ylabel(r'$(\mathcal{I})_{ij}$, [m$^3$]')
    plt.ticklabel_format(style='sci', scilimits=(-4,4), axis='y')



#### HELPER FUNCTIONS ####


def toggle_integral(state):
    with open('Settings/Settings.py', 'r+') as readfile:
        filedata = readfile.read()

    filedata = filedata.replace(f"use_integral = {not state}", f"use_integral = {state}")

    with open('Settings/Settings.py', 'w') as writefile:
        writefile.write(filedata)


if __name__ == '__main__':
    full_suite_of_test_objects(recoverymode=False)
#    hp_refinement_sphere()


