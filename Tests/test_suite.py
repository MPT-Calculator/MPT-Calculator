from pathlib import Path
import sys
import os
from matplotlib import pyplot as plt
import inspect
import numpy as np
from ngsolve import *
#import pytest
try:
    from main import main
    from Functions.Helper_Functions.exact_sphere import *
except:
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    os.chdir(parentdir)
    from main import *
    from Functions.Helper_Functions.exact_sphere import *

"""
James Elgy 2023-2024

This testing module provides some simple checks to make sure that any changes to MPT-Calculator still produce consistent results.
Each test first generates a MPT spectral signature. This signature is then compared with results for the same object and discretisation generated
using the previous release of MPT-Calculator.

Note that this does not make any claim that the results are "accurate", only that they are consistent with the previous version of the software.
Plots of the difference at each frequency are then saved to the Test_Results subdirectory.

In our testing, we use pytest as the testing library, with documentation avaliable at https://docs.pytest.org/en/8.0.x/contents.html#

Paul Ledger 2025

Replaced test_key_4 with a version without a boundary layer due to meshing issues in 6.2.2506 of NGSolve (unable to resolve by healing the geometry)

To run complete test suite: python3 -m pytest -s test_suite.py

To single test example (e.g. test_key):  python3 -m pytest -s test_suite.py::test_key

"""

def test_sphere():
    
    # Running Sweep and computing error
    geometry = 'OCC_test_sphere_prism_32.py'#'OCC_test_sphere_prism_32.geo'#'OCC_test_sphere_prism_32.py'# was order=3 was use_OCC=True added alpha
    test_results = main(geometry=geometry, order=3, use_OCC=True, use_POD=True, use_parallel=False, cpus=4)#, alpha=1e-2)
    test_tensors = test_results['TensorArray'] 
    
    validation_filename = r'Tests/Validation_Standards/OCC_sphere_prism_32/al_0.01_mu_1_sig_1e6/1e1-1e8_40_el_22426_ord_3_POD_13_1e-6/Data'
    valdiation_tensors = np.genfromtxt(validation_filename + '/Tensors.csv', dtype=complex, delimiter=', ')
    
    rel_err = np.zeros(len(test_tensors), dtype=complex)
    for ind in range(len(test_tensors)):
        rel_err[ind] = np.linalg.norm((test_tensors[ind, :] - valdiation_tensors[ind, :])) / np.linalg.norm(valdiation_tensors[ind, :])
    max_err = np.max(rel_err)
    
    
    exactmpt = np.zeros(len(test_results['FrequencyArray']), dtype = complex)

    cnt=0
    for omega in test_results['FrequencyArray']:
        exactmpt[cnt] = exact_sphere(1e-2, 0, 1, 1e6, omega)
        print(omega,exactmpt[cnt])
        cnt+=1

    print(exactmpt)
    print(test_results['TensorArray'][:,0])
    # Generating Comparison Graphs
    plt.close('all')
    
    plt.figure()
    plt.loglog(test_results['FrequencyArray'], rel_err.real)
    plt.xlabel('$\omega$, [rad/s]')
    plt.ylabel('Relative Error')
    plt.savefig('Tests/Test_Results/Sphere_rel_err.pdf')
    
    plt.figure()
    for i in range(9):
        if i == 0:
            plt.semilogx(test_results['FrequencyArray'], test_results['TensorArray'][:,i].real, label='New', color='b')
            plt.semilogx(test_results['FrequencyArray'], valdiation_tensors[:,i].real, label='Standard', color='r')
            plt.semilogx(test_results['FrequencyArray'], exactmpt.real,'mx', label='Exact')
        else:
            plt.semilogx(test_results['FrequencyArray'], test_results['TensorArray'][:,i].real, color='b')
            plt.semilogx(test_results['FrequencyArray'], valdiation_tensors[:,i].real, color='r')
    plt.xlabel('$\omega$, [rad/s]')
    plt.ylabel(r'$(\tilde{\mathcal{R}})_{ij}$, [m$^3$]')
    plt.legend()
    plt.savefig('Tests/Test_Results/Sphere_real.pdf')
    
    plt.figure()
    for i in range(9):
        if i == 0:
            plt.semilogx(test_results['FrequencyArray'], test_results['TensorArray'][:,i].imag, label='New', color='b')
            plt.semilogx(test_results['FrequencyArray'], valdiation_tensors[:,i].imag, label='Standard', color='r')
            plt.semilogx(test_results['FrequencyArray'], exactmpt.imag, 'mx', label='Exact')
        else:
            plt.semilogx(test_results['FrequencyArray'], test_results['TensorArray'][:,i].imag, color='b')
            plt.semilogx(test_results['FrequencyArray'], valdiation_tensors[:,i].imag, color='r')
    plt.xlabel('$\omega$, [rad/s]')
    plt.ylabel(r'$(\mathcal{I})_{ij}$, [m$^3$]')
    plt.legend()
    plt.savefig('Tests/Test_Results/Sphere_imag.pdf')
    
    plt.close('all')
    
    assert max_err < 1e-2
    
def test_magnetic_disk():
    
    # Running Sweep and computing error
    geometry = 'OCC_test_thin_disc_magnetic_32.py'
    test_results = main(geometry=geometry, order=3, use_OCC=True, use_POD=True, alpha=1e-3, use_parallel=False, cpus=4)
    test_tensors = test_results['TensorArray'] 
    
    validation_filename = r'Tests/Validation_Standards/OCC_thin_disc_magnetic_32/al_0.001_mu_32_sig_1e6/1e1-1e8_40_el_27743_ord_3_POD_13_1e-6/Data'
    valdiation_tensors = np.genfromtxt(validation_filename + '/Tensors.csv', dtype=complex, delimiter=', ')
    
    rel_err = np.zeros(len(test_tensors), dtype=complex)
    for ind in range(len(test_tensors)):
        rel_err[ind] = np.linalg.norm((test_tensors[ind, :] - valdiation_tensors[ind, :])) / np.linalg.norm(valdiation_tensors[ind, :])
    max_err = np.max(rel_err)
    
    # Generating Comparison Graphs
    plt.close('all')
    
    plt.figure()
    plt.loglog(test_results['FrequencyArray'], rel_err.real)
    plt.xlabel('$\omega$, [rad/s]')
    plt.ylabel('Relative Error')
    plt.savefig('Tests/Test_Results/Magnetic_Disk_rel_err.pdf')
    
    plt.figure()
    for i in range(9):
        if i == 0:
            plt.semilogx(test_results['FrequencyArray'], test_results['TensorArray'][:,i].real, label='New', color='b')
            plt.semilogx(test_results['FrequencyArray'], valdiation_tensors[:,i].real, label='Standard', color='r')
        else:
            plt.semilogx(test_results['FrequencyArray'], test_results['TensorArray'][:,i].real, color='b')
            plt.semilogx(test_results['FrequencyArray'], valdiation_tensors[:,i].real, color='r')
    plt.xlabel('$\omega$, [rad/s]')
    plt.ylabel(r'$(\tilde{\mathcal{R}})_{ij}$, [m$^3$]')
    plt.legend()
    plt.savefig('Tests/Test_Results/Magnetic_Disk_real.pdf')
    
    plt.figure()
    for i in range(9):
        if i == 0:
            plt.semilogx(test_results['FrequencyArray'], test_results['TensorArray'][:,i].imag, label='New', color='b')
            plt.semilogx(test_results['FrequencyArray'], valdiation_tensors[:,i].imag, label='Standard', color='r')
        else:
            plt.semilogx(test_results['FrequencyArray'], test_results['TensorArray'][:,i].imag, color='b')
            plt.semilogx(test_results['FrequencyArray'], valdiation_tensors[:,i].imag, color='r')
    plt.xlabel('$\omega$, [rad/s]')
    plt.ylabel(r'$(\mathcal{I})_{ij}$, [m$^3$]')
    plt.legend()
    plt.savefig('Tests/Test_Results/Mangetic_Disk_imag.pdf')
    
    plt.close('all')
    
    assert max_err < 1e-2

def test_dualbar():
    
    # Running Sweep and computing error
    geometry = 'OCC_test_dualbar.py'
    test_results = main(geometry=geometry, order=3, use_OCC=True, use_POD=True, alpha=1e-3, use_parallel=False, cpus=4)
    test_tensors = test_results['TensorArray'] 
    
    validation_filename = r'Tests/Validation_Standards/OCC_dualbar/al_0.001_mu_1,1_sig_1e6,1e8/1e1-1e8_40_el_78714_ord_3_POD_13_1e-6/Data'
    valdiation_tensors = np.genfromtxt(validation_filename + '/Tensors.csv', dtype=complex, delimiter=', ')
    
    rel_err = np.zeros(len(test_tensors), dtype=complex)
    for ind in range(len(test_tensors)):
        rel_err[ind] = np.linalg.norm((test_tensors[ind, :] - valdiation_tensors[ind, :])) / np.linalg.norm(valdiation_tensors[ind, :])
    max_err = np.max(rel_err)
    
    # Generating Comparison Graphs
    plt.close('all')
    
    plt.figure()
    plt.loglog(test_results['FrequencyArray'], rel_err.real)
    plt.xlabel('$\omega$, [rad/s]')
    plt.ylabel('Relative Error')
    plt.savefig('Tests/Test_Results/Dualbar_rel_err.pdf')
    
    plt.figure()
    for i in range(9):
        if i == 0:
            plt.semilogx(test_results['FrequencyArray'], test_results['TensorArray'][:,i].real, label='New', color='b')
            plt.semilogx(test_results['FrequencyArray'], valdiation_tensors[:,i].real, label='Standard', color='r')
        else:
            plt.semilogx(test_results['FrequencyArray'], test_results['TensorArray'][:,i].real, color='b')
            plt.semilogx(test_results['FrequencyArray'], valdiation_tensors[:,i].real, color='r')
    plt.xlabel('$\omega$, [rad/s]')
    plt.ylabel(r'$(\tilde{\mathcal{R}})_{ij}$, [m$^3$]')
    plt.legend()
    plt.savefig('Tests/Test_Results/Dualbar_real.pdf')
    
    plt.figure()
    for i in range(9):
        if i == 0:
            plt.semilogx(test_results['FrequencyArray'], test_results['TensorArray'][:,i].imag, label='New', color='b')
            plt.semilogx(test_results['FrequencyArray'], valdiation_tensors[:,i].imag, label='Standard', color='r')
        else:
            plt.semilogx(test_results['FrequencyArray'], test_results['TensorArray'][:,i].imag, color='b')
            plt.semilogx(test_results['FrequencyArray'], valdiation_tensors[:,i].imag, color='r')
    plt.xlabel('$\omega$, [rad/s]')
    plt.ylabel(r'$(\mathcal{I})_{ij}$, [m$^3$]')
    plt.legend()
    plt.savefig('Tests/Test_Results/Dualbar_imag.pdf')
    
    plt.close('all')
    
    assert max_err < 1e-2

def test_key():
    
    # Running Sweep and computing error
    geometry = 'OCC_test_key_4_nomag.py'
    # geometry = 'OCC_test_key_4.py'
    test_results = main(geometry=geometry, order=3, use_OCC=True, use_POD=True, alpha=1e-3, use_parallel=False, cpus=4)
    test_tensors = test_results['TensorArray'] 
    
    #validation_filename = r'Tests/Validation_Standards/OCC_key_4/al_0.001_mu_141.3135696662735_sig_1.5e7/1e1-1e8_40_el_39128_ord_3_POD_13_1e-6/Data'
    validation_filename = r'Tests/Validation_Standards/OCC_key_4_nomag/al_0.001_mu_4_sig_1.5e7/1e1-1e8_40_el_124280_ord_3_POD_13_1e-6/Data'
    
    valdiation_tensors = np.genfromtxt(validation_filename + '/Tensors.csv', dtype=complex, delimiter=', ')
    
    rel_err = np.zeros(len(test_tensors), dtype=complex)
    for ind in range(len(test_tensors)):
        rel_err[ind] = np.linalg.norm((test_tensors[ind, :] - valdiation_tensors[ind, :])) / np.linalg.norm(valdiation_tensors[ind, :])
    max_err = np.max(rel_err)
    
    # Generating Comparison Graphs
    plt.close('all')
    
    plt.figure()
    plt.loglog(test_results['FrequencyArray'], rel_err.real)
    plt.xlabel('$\omega$, [rad/s]')
    plt.ylabel('Relative Error')
    plt.savefig('Tests/Test_Results/Key_rel_err.pdf')
    
    plt.figure()
    for i in range(9):
        if i == 0:
            plt.semilogx(test_results['FrequencyArray'], test_results['TensorArray'][:,i].real, label='New', color='b')
            plt.semilogx(test_results['FrequencyArray'], valdiation_tensors[:,i].real, label='Standard', color='r')
        else:
            plt.semilogx(test_results['FrequencyArray'], test_results['TensorArray'][:,i].real, color='b')
            plt.semilogx(test_results['FrequencyArray'], valdiation_tensors[:,i].real, color='r')
    plt.xlabel('$\omega$, [rad/s]')
    plt.ylabel(r'$(\tilde{\mathcal{R}})_{ij}$, [m$^3$]')
    plt.legend()
    plt.savefig('Tests/Test_Results/Key_real.pdf')
    
    plt.figure()
    for i in range(9):
        if i == 0:
            plt.semilogx(test_results['FrequencyArray'], test_results['TensorArray'][:,i].imag, label='New', color='b')
            plt.semilogx(test_results['FrequencyArray'], valdiation_tensors[:,i].imag, label='Standard', color='r')
        else:
            plt.semilogx(test_results['FrequencyArray'], test_results['TensorArray'][:,i].imag, color='b')
            plt.semilogx(test_results['FrequencyArray'], valdiation_tensors[:,i].imag, color='r')
    plt.xlabel('$\omega$, [rad/s]')
    plt.ylabel(r'$(\mathcal{I})_{ij}$, [m$^3$]')
    plt.legend()
    plt.savefig('Tests/Test_Results/Key_imag.pdf')
    
    plt.close('all')
    
    assert max_err < 1e-2

def test_tetra():
    
    # Running Sweep and computing error
    geometry = 'OCC_test_step_tetra_z5.py'
    test_results = main(geometry=geometry, order=3, use_OCC=True, use_POD=True, alpha=1e-2, use_parallel=False, cpus=4)
    test_tensors = test_results['TensorArray'] 
    
    validation_filename = r'Tests/Validation_Standards/OCC_step_tetra_z5/al_0.01_mu_8_sig_1e6/1e1-1e8_40_el_10240_ord_3_POD_13_1e-6/Data'
    valdiation_tensors = np.genfromtxt(validation_filename + '/Tensors.csv', dtype=complex, delimiter=', ')
    
    rel_err = np.zeros(len(test_tensors), dtype=complex)
    for ind in range(len(test_tensors)):
        rel_err[ind] = np.linalg.norm((test_tensors[ind, :] - valdiation_tensors[ind, :])) / np.linalg.norm(valdiation_tensors[ind, :])
    max_err = np.max(rel_err)
    
    # Generating Comparison Graphs
    plt.close('all')
    
    plt.figure()
    plt.loglog(test_results['FrequencyArray'], rel_err.real)
    plt.xlabel('$\omega$, [rad/s]')
    plt.ylabel('Relative Error')
    plt.savefig('Tests/Test_Results/Tetra_rel_err.pdf')
    
    plt.figure()
    for i in range(9):
        if i == 0:
            plt.semilogx(test_results['FrequencyArray'], test_results['TensorArray'][:,i].real, label='New', color='b')
            plt.semilogx(test_results['FrequencyArray'], valdiation_tensors[:,i].real, label='Standard', color='r')
        else:
            plt.semilogx(test_results['FrequencyArray'], test_results['TensorArray'][:,i].real, color='b')
            plt.semilogx(test_results['FrequencyArray'], valdiation_tensors[:,i].real, color='r')
    plt.xlabel('$\omega$, [rad/s]')
    plt.ylabel(r'$(\tilde{\mathcal{R}})_{ij}$, [m$^3$]')
    plt.legend()
    plt.savefig('Tests/Test_Results/Tetra_real.pdf')
    
    plt.figure()
    for i in range(9):
        if i == 0:
            plt.semilogx(test_results['FrequencyArray'], test_results['TensorArray'][:,i].imag, label='New', color='b')
            plt.semilogx(test_results['FrequencyArray'], valdiation_tensors[:,i].imag, label='Standard', color='r')
        else:
            plt.semilogx(test_results['FrequencyArray'], test_results['TensorArray'][:,i].imag, color='b')
            plt.semilogx(test_results['FrequencyArray'], valdiation_tensors[:,i].imag, color='r')
    plt.xlabel('$\omega$, [rad/s]')
    plt.ylabel(r'$(\mathcal{I})_{ij}$, [m$^3$]')
    plt.legend()
    plt.savefig('Tests/Test_Results/Tetra_imag.pdf')
    
    plt.close('all')
    
    assert max_err < 1e-2


if __name__ == '__main__':
    test_sphere()


