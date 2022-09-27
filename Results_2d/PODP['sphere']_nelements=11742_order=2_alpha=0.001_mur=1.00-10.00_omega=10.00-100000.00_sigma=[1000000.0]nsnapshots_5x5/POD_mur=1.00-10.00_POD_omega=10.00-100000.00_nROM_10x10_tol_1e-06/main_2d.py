"""
James Elgy 2022
Script to run 2D MPT_calculator using the MultiParamSweep class with either RNN, PODP, or PODN
Currently only supports one object. i.e. a single cube.
Parameter names follow the same convention as main.py.
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
from shutil import copyfile



# Flag options about which method to use for the ROM modelling.
PODP = True
PODN = False
NNR = False


# Object Settings

# Vol file mesh name. e.g. 'sphere.vol'
Mesh_Name = 'OCC_sphere.vol'

# Alpha scaling term as float in m.
Alpha = 0.001
# (float) scaling to be applied to the .vol file i.e. if you have defined
# a sphere of unit radius in a .geo file   alpha = 0.01   would simulate a
# sphere with a radius of 0.01m ( or 1cm)

# The order of the elements in the mesh
Order = 2
# (int) this defines the order of each of the elements in the mesh

# List of conductivites for each object in the mesh.
Conductivity = [1e6]
# (list) this defines an ordered list of conductivities for each object in the mesh. e.g. if you have two objects sphere
# and box, then [1e6, 2e5] would give the sphere 1e6 and the box 2e5.


# Sweep Settings:

# About the Frequency sweep (frequencies are in radians per second)
# Minimum frequency (Powers of 10 i.e Start = 2 => 10**2)
Start = 1

# Maximum frequency (Powers of 10 i.e Start = 8 => 10**8)
Finish = 5

# (int) Number of logarithmically spaced points in the frequency sweep
Points = 10

# Equivalent mur settings. These are stored linearly. i.e. a Mur_Finish of 13 would give an upper bound of mur=13.
Mur_Start = 1
Mur_Finish = 10
Mur_Points = 10

# ROM Settings:

# Number of frequency snapshots used in the ROM.
N_Snapshots = 5
# (int) number of frequencies to use when generating the ROM. Typically 16 is suffieicnet.

# Number of mur values to use in the ROM
Mur_N_Snapshots = 5
# (int) number of mur snapshots to use for the 2D ROM.

# Truncation tolerances for the SVD. Smaller provides more accuracy for greater computational expence.
Theta0_Tol = 1e-6
Theta1_Tol = 1e-6
# (float) Truncation tolerances for the SVD used in the PODN and PODP schemes.

#### RUN THE SCRIPT ####

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
MPS.permeability_array_ROM = np.linspace(Mur_Start, Mur_Finish, Mur_Points)
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

copyfile('main_2d.py', foldername+'/main_2d.py')
copyfile('Settings/Settings.py', foldername+'/Settings.py')


print('Done')


