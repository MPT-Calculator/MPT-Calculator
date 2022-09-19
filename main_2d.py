"""
James Elgy 2022
Script to run 2D MPT_calculator using the MultiParamSweep class with either RNN, PODP, or PODN
"""


import sys
import numpy as np
import os

sys.path.insert(0, "Functions")
sys.path.insert(0, "Settings")

from MultiPermeability import *
from Settings import *
from ML_MPT_Predictor import ML, DataLoader
import pandas as pd

PODP = True
PODN = False
NNR = False

# Object Settings
Mesh_Name = 'sphere.vol'
Alpha = 0.001
Order = 2
Conductivity = [1e6]

# Sweep Settings:
Start = 1
Finish = 5
Points = 32
Mur_Start = 1
Mur_Finish = 50
Mur_Points = 32

# ROM Settings:
N_Snapshots = 16
Mur_N_Snapshots = 16
Theta0_Tol = 1e-6
Theta1_Tol = 1e-6

PlotPod, PODErrorBars, EddyCurrentTest, vtk_output, Refine_vtk = AdditionalOutputs()

# Loading Mesh:
ngmesh = ngmeshing.Mesh(dim=3)
ngmesh.Load("VolFiles/" + Mesh_Name)
mesh = Mesh("VolFiles/" + Mesh_Name)

# Construct dictionary of conductivities.
matlist = []
with open("VolFiles/" + Mesh_Name, 'r') as f:
    lines = f.readlines()
    for i in range(0, len(lines)):
        line = lines[i]
        if line[:9] == 'materials':
            n_mats = int(lines[i + 1])
            for j in range(n_mats):
                matlist += [lines[i + j + 2][2:-1]]
            break
matlist = list(set(matlist))
Conductivity_Dict = {}
counter = 0
for mat in matlist:
    if mat != 'air':
        Conductivity_Dict.update({mat: Conductivity[counter]})
        counter += 1
    else:
        Conductivity_Dict.update({'air': 0})

# Initalise the MultiParamSweep class for constructing 2d characterisations.
MPS = MultiParamSweep(Alpha, Order, Conductivity_Dict, mesh)
MPS.frequency_array = np.logspace(Start, Finish, N_Snapshots)
MPS.mur_max = Mur_Finish
MPS.mur_min = Mur_Start
MPS.mur_points = Mur_N_Snapshots
MPS.permeability_array_ROM = np.logspace(Mur_Start, Mur_Finish, Mur_Points)
MPS.frequency_array_ROM = np.logspace(Start, Finish, Points)
MPS.generate_new_snapshot_mur_positions()

# Running Sweep:
MPS.calc_theta0_snapshots(apply_postprojection=True)
MPS.calc_theta1_snapshots(apply_postprojection=False)

MPS.calc_N0_snapshots()
MPS.calc_R_snapshots()
MPS.calc_I_snapshots()

if PODP is True:
    foldername = MPS.save_results(prefix='PODP')
    PODP = MPS.PODP_ROM(calc_error_bars=PODErrorBars)
    MPS.calc_I_ROM()
    MPS.calc_R_ROM()
    MPS.calc_N0_ROM()
    MPS.calc_eigenvalues(use_ROM=True)

    MPS.plot_N0(use_ROM=True)
    MPS.plot_real_component(use_ROM=True, style='surf', plot_errorbars=PODErrorBars)
    MPS.plot_imag_component(use_ROM=True, style='surf', plot_errorbars=PODErrorBars)
    MPS.save_results(prefix='PODP')

if NNR is True:
    foldername = MPS.save_results(prefix='NNR')
    eig_snap = np.zeros((3, len(MPS.permeability_array), len(MPS.frequency_array)), dtype=complex)
    for ind in range(len(MPS.permeability_array)):
        for jnd in range(len(MPS.frequency_array)):
            eig_snap[:, ind, jnd] = np.linalg.eigvals(
                MPS.R[:, :, ind, jnd] + MPS.N0_snapshots[ind, :, :]) + 1j * np.linalg.eigvals(MPS.I[:, :, ind, jnd])

    np.save(foldername + '/eigenvalues_snap.npy', eig_snap)
    # POD_RNN
    data = pd.DataFrame({'omega': [],
                         'mur': [],
                         'eig_1_real': [], 'eig_2_real': [], 'eig_3_real': [],
                         'eig_1_imag': [], 'eig_1_imag': [], 'eig_1_imag': [],
                         'N0': [], 'tensor_coeffs': []})

    for ind, mur in enumerate(MPS.permeability_array):
        EigenValues_mur = np.squeeze(eig_snap[:, ind, :]).transpose()
        eig_1_real = EigenValues_mur[:, 0].real
        eig_2_real = EigenValues_mur[:, 1].real
        eig_3_real = EigenValues_mur[:, 2].real
        eig_1_imag = EigenValues_mur[:, 0].imag
        eig_2_imag = EigenValues_mur[:, 1].imag
        eig_3_imag = EigenValues_mur[:, 2].imag

        omega = MPS.frequency_array

        append_data = pd.DataFrame({'omega': omega,
                                    'mur': [mur] * len(MPS.frequency_array),
                                    'eig_1_real': eig_1_real, 'eig_2_real': eig_2_real, 'eig_3_real': eig_3_real,
                                    'eig_1_imag': eig_1_imag, 'eig_2_imag': eig_2_imag, 'eig_3_imag': eig_3_imag})
        data = data.append(append_data, ignore_index=True)

    data.to_csv(foldername + '/fulldataset.csv')
    D = DataLoader()
    data = D.load_data(foldername + '/fulldataset.csv')
    D.train_test_split(0.0)

    for component in ['eig_3_real', 'eig_2_real', 'eig_1_real']:
        scaling, data = D.preprocess_data()
        ml = ML(D, component)
        ml.perform_fit((8, 8))
        output = ml.postprocess_data()
        fig, zz_real = ml.generate_surface(MPS.frequency_array_ROM, MPS.permeability_array_ROM)

    for component in ['eig_3_imag', 'eig_2_imag', 'eig_1_imag']:
        scaling, data = D.preprocess_data()
        ml = ML(D, component)
        ml.perform_fit((8, 8))
        output = ml.postprocess_data()
        _, zz_imag = ml.generate_surface(MPS.frequency_array_ROM, MPS.permeability_array_ROM)
    eig_NNR = zz_real + 1j * zz_imag
    foldername = MPS.save_results()
    np.save(foldername + '/eigenvalues_RNN.npy', eig_NNR)

if PODN is True:
    foldername = MPS.save_results(prefix='PODN')
    MPS.calc_theta0_ROM()
    MPS.calc_theta1_ROM()

    MPS.calc_I_ROM()
    MPS.calc_R_ROM()
    MPS.calc_N0_ROM()
    MPS.calc_eigenvalues(use_ROM=True)

    MPS.plot_N0(use_ROM=True)
    MPS.plot_real_component(use_ROM=True, style='surf', plot_errorbars=PODErrorBars)
    MPS.plot_imag_component(use_ROM=True, style='surf', plot_errorbars=PODErrorBars)
    MPS.save_results(prefix='PODN')

print('Done')


