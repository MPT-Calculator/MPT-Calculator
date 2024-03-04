from pathlib import Path
import sys
import os
from matplotlib import pyplot as plt

import numpy as np
from main import main

import pytest

def test_sphere():
    
    # Running Sweep and computing error
    geometry = 'OCC_test_sphere_prism_32.py'
    test_results = main(geometry=geometry, order=3, use_OCC=True, use_POD=True)
    test_tensors = test_results['TensorArray'] 
    
    validation_filename = r'Tests/Validation_Standards/OCC_sphere_prism_32/al_0.01_mu_1_sig_1e6/1e1-1e8_40_el_22426_ord_3_POD_13_1e-6/Data'
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
    plt.savefig('Tests/Test_Results/Sphere_rel_err.pdf')
    
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
    plt.savefig('Tests/Test_Results/Sphere_real.pdf')
    
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
    plt.savefig('Tests/Test_Results/Sphere_imag.pdf')
    
    plt.close('all')
    
    assert max_err < 1e-2
    
def test_magnetic_disk():
    
    # Running Sweep and computing error
    geometry = 'OCC_test_thin_disc_magnetic_32.py'
    test_results = main(geometry=geometry, order=3, use_OCC=True, use_POD=True, alpha=1e-3)
    test_tensors = test_results['TensorArray'] 
    
    validation_filename = r'Tests/Validation_Standards/OCC_thin_disc_magnetic_32/al_0.001_mu_32_sig_1e6\1e1-1e8_40_el_27743_ord_3_POD_13_1e-6/Data'
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
    test_results = main(geometry=geometry, order=3, use_OCC=True, use_POD=True, alpha=1e-3)
    test_tensors = test_results['TensorArray'] 
    
    validation_filename = r'Tests\Validation_Standards\OCC_dualbar\al_0.001_mu_1,1_sig_1e6,1e8\1e1-1e8_40_el_78714_ord_3_POD_13_1e-6\Data'
    valdiation_tensors = np.genfromtxt(validation_filename + '\\Tensors.csv', dtype=complex, delimiter=', ')
    
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
    geometry = 'OCC_test_key_4.py'
    test_results = main(geometry=geometry, order=3, use_OCC=True, use_POD=True, alpha=1e-3)
    test_tensors = test_results['TensorArray'] 
    
    validation_filename = r'Tests\Validation_Standards\OCC_key_4\al_0.001_mu_141.3135696662735_sig_1.5e7\1e1-1e8_40_el_39128_ord_3_POD_13_1e-6\Data'
    valdiation_tensors = np.genfromtxt(validation_filename + '\\Tensors.csv', dtype=complex, delimiter=', ')
    
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
    test_results = main(geometry=geometry, order=3, use_OCC=True, use_POD=True, alpha=1e-2)
    test_tensors = test_results['TensorArray'] 
    
    validation_filename = r'Tests\Validation_Standards\OCC_step_tetra_z5\al_0.01_mu_8_sig_1e6\1e1-1e8_40_el_10240_ord_3_POD_13_1e-6\Data'
    valdiation_tensors = np.genfromtxt(validation_filename + '\\Tensors.csv', dtype=complex, delimiter=', ')
    
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


