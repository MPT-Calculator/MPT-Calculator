"""
James Elgy - 2022
Module for the calculation of the MPT for multiple permeabilities.

"""
import os
import sys
os.environ['OMP NUM THREADS'] = '1'
os.environ['MKL NUM THREADS'] = '1'
os.environ['MKL THREADING LAYER'] = 'sequential'

import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + '/GeoFiles')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + '/Settings')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + '/JamesAddons')
from Functions.Old_versions.MPTFunctions import *
from main import main as direct_solver

import multiprocessing_on_dill as multiprocessing
import Settings
import netgen.meshing as ngmeshing
from matplotlib import pyplot as plt
from sklearn import neural_network as nn
from sklearn import preprocessing as pp
from sklearn.model_selection import GridSearchCV
import scipy.sparse as sp
import numpy as np
import pickle
import tikzplotlib
import pandas as pd
import seaborn as sns
# import tkinter.filedialog  # Used for file dialog in figure loader.
# from tkinter import Tk
import time
from ML_MPT_Predictor import ML, DataLoader


class MultiParamSweep():
    """
    James Elgy 2022
    This class is designed to enable the calculation of MPT spectral signatures as a fuction of both frequency and
    permeability. To this end, it employs a neural network based approach to reduced order modelling. The class is
    initiated with a pregenerated mesh, order, alpha, and dictionary of conductivities.

    The variables permeability_array and frequency_array set the snapshot parameters, while permeability_array_ROM and
    frequency_array_ROM set the queried parameters.
    """

    def __init__(self, alpha, order, sigma_dict, mesh):
        # Solver parameters:
        self.Solver, self.epsi, self.Maxsteps, self.Tolerance = Settings.SolverParameters()

        # Object parameters:
        self.mesh = mesh
        self.object_name = [mat for mat in mesh.GetMaterials() if mat!='air'][0]
        self.mesh.Curve(5)
        self.alpha = alpha
        self.order = order
        self.object_list = self.mesh.GetMaterials()
        inorout = dict(zip(self.object_list, [0 if mat == 'air' else 1 for mat in self.mesh.GetMaterials()]))
        inout_coef = [inorout[mat] for mat in mesh.GetMaterials()]
        self.inout = CoefficientFunction(inout_coef)
        self.sigma_dict = sigma_dict
        # fes = HCurl(mesh, order=Order, dirichlet="outer", flags={"nograds": True})
        self.evec = [CoefficientFunction((1, 0, 0)), CoefficientFunction((0, 1, 0)), CoefficientFunction((0, 0, 1))]
        self.xivec = [CoefficientFunction((0, -z, y)), CoefficientFunction((z, 0, -x)), CoefficientFunction((-y, x, 0))]

        # Snapshot parameters:
        self.frequency_array = np.logspace(1, 5, 16)
        self.log_base_frequency = 10

        self.mur_max = 50
        self.mur_min = 1
        self.mur_points = 16

        min_inv_mur = 1/self.mur_max
        max_inv_mur = 1/self.mur_min
        log_base = self.mur_max**(1/np.log10(max(self.frequency_array)))
        N_Samples = self.mur_points
        Delta = (np.log(max_inv_mur)/np.log(log_base)) - (np.log(min_inv_mur)/np.log(log_base))
        Delta = Delta / (N_Samples-1)
        s = np.log(max_inv_mur)/np.log(log_base)
        self.permeability_array = [log_base**s]

        for ind in range(N_Samples-1):
            s += Delta
            self.permeability_array += [log_base**s]
        self.permeability_array = np.asarray(self.permeability_array)
        self.log_base_permeability = log_base
        # single permeability sweep:
        # self.permeability_array = [10]
        # self.log_base_permeability = 10


        # Neural net parameters:
        self.permeability_array_ROM = np.logspace(0,np.log10(50),32)
        self.frequency_array_ROM = np.logspace(1,5,32)
        self.theta0_truncation_tol = 1e-6
        self.theta1_truncation_tol = 1e-6

    def generate_new_snapshot_mur_positions(self):
        min_inv_mur = 1 / self.mur_max
        max_inv_mur = 1 / self.mur_min
        log_base = self.mur_max ** (1 / np.log10(max(self.frequency_array)))
        N_Samples = self.mur_points
        Delta = (np.log(max_inv_mur) / np.log(log_base)) - (np.log(min_inv_mur) / np.log(log_base))
        Delta = Delta / (N_Samples - 1)
        s = np.log(max_inv_mur) / np.log(log_base)
        self.permeability_array = [log_base ** s]

        for ind in range(N_Samples - 1):
            s += Delta
            self.permeability_array += [log_base ** s]
        self.permeability_array = np.asarray(self.permeability_array)
        self.log_base_permeability = log_base

    def calc_theta0_snapshots(self, apply_postprojection=True, parallel=True, sparse=False):
        """
        Function to generate theta0 solutions given permeability array, solver, and material properties.
        Appends solutions and finite element space to self.
        :return:
        """

        print('Solving Theta0 Problem')
        # Using gradients inside the object makes the theta0 solutions easier to work with when also considering the
        # theta1 solutions since they have the same basis functions.
        dom_nrs_metal = [0 if mat == "air" else 1 for mat in self.mesh.GetMaterials()]
        theta0_fes = HCurl(self.mesh, order=self.order, dirichlet='outer', gradientdomains=dom_nrs_metal)#, flags = { "nograds" : True })
        # theta0_fes = HCurl(self.mesh, order=self.order, dirichlet='outer', flags = { "nograds" : True })
        theta0_fes_complex = HCurl(self.mesh, order=self.order, dirichlet='outer', complex=True, gradientdomains=dom_nrs_metal)#, flags = { "nograds" : True }) # Identical to theta1_fes.

        # theta0_fes = HCurl(self.mesh, order=self.order,dirichlet="outer", flags={"nograds": True})
        # theta0_fes_complex = HCurl(self.mesh, order=self.order, dirichlet="outer", flags={"nograds": True}, complex=True)

        D0_full = np.zeros((theta0_fes.ndof, len(self.permeability_array), 3))

        self.theta0_fes = theta0_fes
        self.theta0_fes_complex = theta0_fes_complex

        for dim in [0, 1, 2]:  # For each dimension in evec calculate theta0 and store.
            input = []
            for ind, mu in enumerate(self.permeability_array):
                mur = dict(zip(self.object_list, [1 if mat == 'air' else mu for mat in self.mesh.GetMaterials()]))
                mur_coeff = CoefficientFunction([mur[mat] for mat in self.mesh.GetMaterials()])
                new_input = (theta0_fes, self.order, self.alpha, mur_coeff, self.inout, self.evec[dim],
                             self.Tolerance, self.Maxsteps, self.epsi, dim, self.Solver)
                input.append(new_input)

            if parallel == True:
                with multiprocessing.Pool(
                        processes=4) as pool:
                    output = pool.starmap(Theta0, input)

                for ind, mu in enumerate(self.permeability_array):
                    D0_full[:, ind, dim] = output[ind]

            else:
                for ind, mu in enumerate(self.permeability_array):
                    mur = dict(zip(self.object_list, [1 if mat == 'air' else mu for mat in self.mesh.GetMaterials()]))
                    mur_coeff = CoefficientFunction([mur[mat] for mat in self.mesh.GetMaterials()])
                    D0_full[:, ind, dim] = Theta0(theta0_fes, self.order, self.alpha, mur_coeff, self.inout, self.evec[dim],
                                 self.Tolerance, self.Maxsteps, self.epsi, dim+1, self.Solver)


        if apply_postprojection == True:
            print('Applying Postprocessing')
            # Postprocessing to set appropriate gauge. Follows Zaglmayr.
            proj = self._define_postprojection()
            for dim in range(3):
                theta0 = GridFunction(theta0_fes)
                for ind in range(len(self.permeability_array)):
                    theta0.vec.FV().NumPy()[:] = D0_full[:, ind, dim]
                    theta0.vec.data = proj * (theta0.vec)
                    D0_full[:, ind, dim] = theta0.vec.FV().NumPy()[:]
            # self.projection = proj
        self.theta0_fes = theta0_fes
        if sparse is True:
            self.theta0_snapshots = [sp.csr_matrix(D0_full[:,:,0].real), sp.csr_matrix(D0_full[:,:,1].real), sp.csr_matrix(D0_full[:,:,2].real)]
        else:
            self.theta0_snapshots = D0_full.real

    def calc_theta1_snapshots(self, apply_postprojection=False, parallel=True, sparse=False):
        """
        Function to generate theta1 solutions given permeability array, solver, and material properties.
        Appends solutions and finite element space to self.
        Calls Ben's Theta1 function in the multiprocessing section of code.
        :return:
        """

        print('Solving Theta1 Problem')
        dom_nrs_metal = [0 if mat == "air" else 1 for mat in self.mesh.GetMaterials()]
        theta1_fes = HCurl(self.mesh, order=self.order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
        theta0_fes = self.theta0_fes_complex
        xivec = [CoefficientFunction((0, -z, y)), CoefficientFunction((z, 0, -x)), CoefficientFunction((-y, x, 0))]  # xivec is cross product between identity and [x,y,z]
        D1_full = np.zeros((theta1_fes.ndof, 3, len(self.permeability_array), len(self.frequency_array)), dtype=complex)

        input = []
        sig_coeff = CoefficientFunction([self.sigma_dict[mat] for mat in self.mesh.GetMaterials()])
        for dim in range(3):
            for ind, mu in enumerate(self.permeability_array):
                for jnd, omega in enumerate(self.frequency_array):
                    nu = omega * 4 * np.pi * 1e-7 * (self.alpha ** 2)
                    mur = dict(zip(self.object_list, [1 if mat == 'air' else mu for mat in self.mesh.GetMaterials()]))
                    mur_coeff = CoefficientFunction([mur[mat] for mat in self.mesh.GetMaterials()])
                    if sparse is True:
                        D0 = self.theta0_snapshots[dim][:,[ind]] + (1j*np.zeros(self.theta0_snapshots[dim][:,[ind]].shape))
                    else:
                        D0 = self.theta0_snapshots[:,ind,dim] + (1j*np.zeros(self.theta0_snapshots[:,ind,dim].shape))
                    new_input = (
                        theta0_fes, theta1_fes, D0, xivec[dim], self.order, self.alpha, nu, sig_coeff, mur_coeff, self.inout,
                        self.Tolerance, self.Maxsteps, self.epsi, omega, dim+1, 3, self.Solver)
                    input.append(new_input)

        if parallel == True:
            # For simplicity, this calls Theta1 rather than Theta1_Sweep. The code would probably run quicker if it used
            # Theta1_Sweep.
            with multiprocessing.Pool(processes=4) as pool:
                output = pool.starmap(Theta1, input)

            output = np.asarray(output)

            # Unpacking output.
            count = 0
            for dim in range(3):
                for ind, mu in enumerate(self.permeability_array):
                    for jnd, omega in enumerate(self.frequency_array):
                        D1_full[:, dim, ind, jnd] = output[count, :]
                        count += 1

        else:
            for dim in range(3):
                for ind, mu in enumerate(self.permeability_array):
                    for jnd, omega in enumerate(self.frequency_array):
                        nu = omega * 4 * np.pi * 1e-7 * (self.alpha ** 2)
                        mur = dict(zip(self.object_list, [1 if mat == 'air' else mu for mat in self.mesh.GetMaterials()]))
                        mur_coeff = CoefficientFunction([mur[mat] for mat in self.mesh.GetMaterials()])
                        D0 = self.theta0_snapshots[:, ind, dim] + (1j * np.zeros(self.theta0_snapshots[:, ind, dim].shape))
                        D1_full[:,dim,ind,jnd] = Theta1(theta0_fes, theta1_fes, D0, xivec[dim], self.order, self.alpha, nu, sig_coeff, mur_coeff, self.inout,
                        self.Tolerance, self.Maxsteps, self.epsi, omega, dim+1, 3, self.Solver)


        if apply_postprojection == True:
            for dim in range(3):
                theta1 = GridFunction(theta1_fes)
                for ind in range(len(self.permeability_array)):
                    for jnd in range(len(self.frequency_array)):
                        theta1.vec.FV().NumPy()[:] = D1_full[:, dim ,ind, jnd]
                        theta1.vec.data = self.projection * (theta1.vec)
                        D1_full[:, dim, ind, jnd] = theta1.vec.FV().NumPy()[:]

        self.theta1_fes = theta1_fes
        if sparse is True:
            self.theta1_snapshots = sp.csr_matrix(D1_full)
        else:
            self.theta1_snapshots = D1_full

    def calc_theta0_ROM(self, plot_singular_values=True, return_models=True, return_SVD=True, sparse=False):
        """
        James Elgy - 2022.
        Function to interpolate theta0 snapshots to a finer sampling.
        Uses truncated SVD to obtain U^m @ Sig^m @ (V^m)^H approx D. Neural network is then used to approximate
        (V^m)^H for each retained mode and direction.
        :return:
        """
        x_array = np.log10(self.permeability_array)/np.log10(self.log_base_permeability) # log10 has been shown to perform better than the original coordinates.
        x_query = np.log10(self.permeability_array_ROM)/np.log10(self.log_base_permeability)
        D_pred = np.zeros((self.theta0_fes.ndof, len(self.permeability_array_ROM), 3))

        try:
            fig_num = plt.get_fignums()[-1] + 1
        except:
            fig_num = 1

        if plot_singular_values is True:
            plt.figure(fig_num)

        if return_models is True:
            model_cube = []

        if return_SVD is True:
            self.theta0_U_truncated_array = []
            self.theta0_Vh_truncated_array = []
            self.theta0_Sigma_truncated_array = []

        for dim in range(3):

            if return_models is True:
                model_array = []

            # if sparse is true we temporarily convert to dense matrix such that we can do the full svd and truncate
            # at a specific tolerance.
            if sparse is True:
                U_trunc, S_trunc, Vh_trunc = self._truncated_SVD(self.theta0_snapshots[dim].todense(),
                                                                 tol=self.theta0_truncation_tol)
                n_modes = len(np.diag(S_trunc))
            else:
                U_trunc, S_trunc, Vh_trunc = self._truncated_SVD(self.theta0_snapshots[:,:,dim], tol=self.theta0_truncation_tol)
                n_modes = len(np.diag(S_trunc))
            if return_SVD is True:
                self.theta0_U_truncated_array.append(U_trunc)
                self.theta0_Vh_truncated_array.append(Vh_trunc)
                self.theta0_Sigma_truncated_array.append(S_trunc)

            # plotting decaying singular values.
            if plot_singular_values is True:
                plt.figure(fig_num)
                plt.semilogy(np.diag(S_trunc)/S_trunc[0,0], marker='x', label=f'dir={dim}')
                plt.ylabel('Normalised Singular Values')
                plt.xlabel('Mode')
                plt.show()

            Vh_pred = np.zeros((n_modes, len(x_query)))
            for mode in range(n_modes): # For each retained mode, use neural network to approximate the responce at query.
                y_array = Vh_trunc[mode, :]
                NN = NeuralNetwork(x_array.reshape((len(x_array), 1)),
                                   y_array.reshape((len(y_array), 1)),
                                   x_query.reshape((len(x_query), 1)))
                Vh_pred[mode, :] = NN.run()

                if return_models is True:
                    r_i_models = [NN]
                    model_array.append(r_i_models)

            if return_models is True:
                model_cube.append(model_array)

            D_pred[:,:,dim] = U_trunc @ S_trunc @ Vh_pred
            # self._plot_modes(Vh_trunc, Vh_pred)

        if return_models is True:
            self.theta0_NN_models = model_cube

        if sparse is True:
            self.theta0_ROM = sp.csr_array(D_pred)
        else:
            self.theta0_ROM = D_pred

    def calc_theta1_ROM(self, plot_modes=False, plot_singular_values=True, return_models=False, return_SVD=True, sparse=False):
        """
        James Elgy - 2022
        Function to interpolate the theta1 snapshots in the same manner as theta0_ROM.
        Uses truncated SVD to obtain U^m @ Sig^m @ (V^m)^H approx D. Neural network is then used to approximate
        (V^m)^H for each retained mode and direction. THis is done separately for both real and imaginary components.

        Note that this is a 2d problem.
        :return:
        """

        if return_models is True:
            model_list = []

        x_array = np.log10(self.permeability_array)/np.log10(self.log_base_permeability) # log10 has been shown to perform better than original array.
        y_array = np.log10(self.frequency_array)
        x_query = np.log10(self.permeability_array_ROM)/np.log10(self.log_base_permeability)
        y_query = np.log10(self.frequency_array_ROM)

        xx,yy = np.meshgrid(x_array, y_array)
        plt.figure(); plt.scatter(xx, yy)

        # reshaping theta1 solutions into ndofx3xn^2 array.
        if sparse is False:
                theta1_solutions = np.reshape(self.theta1_snapshots, (self.theta1_fes.ndof, 3, len(self.permeability_array) * len(self.frequency_array)))
        # theta1_pred = np.zeros(self.theta1_snapshots.shape, dtype=complex)
        try:
            fig_num = plt.get_fignums()[-1] + 1
        except:
            fig_num = 1

        if plot_singular_values is True:
            plt.figure(fig_num) # For plotting sigma decay.

        D_pred = np.zeros((self.theta0_fes.ndof, 3, len(self.permeability_array_ROM)*len(self.frequency_array_ROM)), dtype=complex)

        if return_models is True:
            model_cube = []

        if return_SVD is True:
            self.theta1_U_truncated_array = []
            self.theta1_Vh_truncated_array = []
            self.theta1_Sigma_truncated_array = []

        for dim in range(3):

            # if sparse is true we temporarily convert to dense matrix such that we can do the full svd and truncate
            # at a specific tolerance.
            if sparse is True:
                # unpacking theta1 snapshots:
                theta1_solutions = np.empty((self.theta1_fes.ndof, 3, len(self.permeability_array) * len(self.frequency_array)))
                t1_dim = self.theta1_snapshots[dim]
                for ind in range(len(self.permeability_array)):
                    sparse_matrix = t1_dim[ind].todense()
                    theta1_solutions[:,ind, :] = sparse_matrix

                theta1_solutions = np.reshape(theta1_solutions, (self.theta1_fes.ndof,len(self.permeability_array) * len(self.frequency_array)))
                U_trunc, S_trunc, Vh_trunc = self._truncated_SVD(theta1_solutions,
                                                                 tol=self.theta0_truncation_tol)
                n_modes = len(np.diag(S_trunc))
            else:
                U_trunc, S_trunc, Vh_trunc = self._truncated_SVD(theta1_solutions[:, dim, :], tol=self.theta1_truncation_tol)
                n_modes = len(np.diag(S_trunc))

            if return_SVD is True:
                self.theta1_U_truncated_array.append(U_trunc)
                self.theta1_Vh_truncated_array.append(Vh_trunc)
                self.theta1_Sigma_truncated_array.append(S_trunc)

            # plotting sigma decay.
            if plot_singular_values is True:
                plt.figure(fig_num)
                plt.semilogy(np.diag(S_trunc)/S_trunc[0,0], marker='x', label=f'dim={dim}')
                plt.xlabel('Mode')
                plt.ylabel('Normalised $\Sigma')
                plt.legend()

            real_Vh = Vh_trunc.real
            imag_Vh = Vh_trunc.imag
            real_Vh_pred = np.zeros((Vh_trunc.shape[0], len(self.frequency_array_ROM) * len(self.permeability_array_ROM)))
            imag_Vh_pred = np.zeros((Vh_trunc.shape[0], len(self.frequency_array_ROM) * len(self.permeability_array_ROM)))
            Vh_pred = np.zeros(imag_Vh_pred.shape, dtype=complex)

            if return_models is True:
                model_array = []

            for mode in range(n_modes):


                real_mode = real_Vh[mode, :]
                imag_mode = imag_Vh[mode, :]

                # Real and imaginary components are fitted seperatly.
                NN_real = NeuralNetwork(
                                   x_array.reshape((len(x_array), 1)),
                                   y_array.reshape((len(y_array), 1)),
                                   x_query.reshape((len(x_query), 1)),
                                   y_query=y_query.reshape((len(y_query), 1)),
                                   z_array=real_mode.reshape((len(real_mode), 1)))

                NN_imag = NeuralNetwork(
                                   x_array.reshape((len(x_array), 1)),
                                   y_array.reshape((len(y_array), 1)),
                                   x_query.reshape((len(x_query), 1)),
                                   y_query=y_query.reshape((len(y_query), 1)),
                                   z_array=imag_mode.reshape((len(imag_mode), 1)))

                real_Vh_pred[mode, :] = NN_real.run()
                imag_Vh_pred[mode, :] = NN_imag.run()

                if return_models is True:
                    r_i_models = [NN_real, NN_imag]
                    model_array.append(r_i_models)

                Vh_pred[mode,:] = real_Vh_pred[mode,:] + 1j*imag_Vh_pred[mode,:]

            if return_models is True:
                model_cube.append(model_array)

            D_pred[:, dim, :] = U_trunc @ S_trunc @ Vh_pred

            if plot_modes == True:
                self._plot_modes(real_Vh, real_Vh_pred)

        if sparse is True:
            self.theta1_ROM = sp.coo_array(np.reshape(D_pred, (self.theta1_fes.ndof, 3, len(self.permeability_array_ROM), len(self.frequency_array_ROM)))
                                           )
        else:
            self.theta1_ROM = np.reshape(D_pred, (self.theta1_fes.ndof, 3, len(self.permeability_array_ROM), len(self.frequency_array_ROM)))

        if return_models is True:
            self.theta1_NN_models = model_cube

        store_SVD = False
        if store_SVD:
            self.theta1_S = S_trunc
            self.theta1_U = U_trunc
            self.theta1_Vh = Vh_trunc
            self.theta1_Vh_pred = Vh_pred

    def PODP_ROM(self, calc_error_bars=False):
        PODP = PODP_ROM(self)
        PODP.calc_ROM()
        if calc_error_bars is True:
            PODP.calc_error_bars_2()

        # self.theta1_U_truncated_array = [PODP.u1Truncated, PODP.u2Truncated, PODP.u3Truncated]
        # self.theta1_Sigma_truncated_array = [PODP.sigma1Truncated, PODP.sigma2Truncated, PODP.sigma3Truncated]
        # self.theta0_Sigma_truncated_array = [PODP.sigma1Truncated_theta0, PODP.sigma2Truncated_theta0, PODP.sigma3Truncated_theta0]
        # self.theta0_U_truncated_array = [PODP.u1Truncated_theta0, PODP.u2Truncated_theta0, PODP.u3Truncated_theta0]

        return PODP

    def calc_N0_snapshots(self):
        """
        James Elgy - 2022
        Function to calculate N0 given theta0 snapshots for each relative permeability.
        Function also calculates the exact N0 for a sphere for testing purposes.
        :return: N0 and N0_exact.
        """

        N0_array = []
        N0_exact_array = []

        for ind, mu in enumerate(self.permeability_array):  # Assigning permeability
            print(f'Calculating N0 - mur = {mu} - ind = {ind}')

            Theta0i = GridFunction(self.theta0_fes_complex)
            Theta0j = GridFunction(self.theta0_fes_complex)
            N0 = np.zeros((3, 3), dtype=complex)

            mur = dict(zip(self.object_list, [1 if mat == 'air' else mu for mat in self.mesh.GetMaterials()]))

            mur_coeff = CoefficientFunction([mur[mat] for mat in self.mesh.GetMaterials()])
            # Theta0Sol = np.tile(D[:,ind],(3,1)).transpose()  # For sphere all directions give same result.
            Theta0Sol = self.theta0_snapshots[:, ind, :] + (1j*np.zeros(self.theta0_snapshots[:, ind, :].shape))

            # Calculate the N0 tensor
            VolConstant = Integrate(1 - mur_coeff ** (-1), self.mesh)
            for i in range(3):
                Theta0i.vec.FV().NumPy()[:] = Theta0Sol[:, i]
                for j in range(i+1):
                    Theta0j.vec.FV().NumPy()[:] = Theta0Sol[:, j]
                    if i == j:
                        N0[i, j] = (self.alpha ** 3) * (VolConstant + (1 / 4) * (
                            Integrate(mur_coeff ** (-1) * (InnerProduct(curl(Theta0i), curl(Theta0j))), self.mesh)))
                    else:
                        N0[i, j] = (self.alpha ** 3 / 4) * (
                            Integrate(mur_coeff ** (-1) * (InnerProduct(curl(Theta0i), curl(Theta0j))), self.mesh))

            # Copy the tensor
            N0 += np.transpose(N0 - np.diag(np.diag(N0)))
            N0_array += [N0]

            # Calc exact N0 for sphere
            N0_exact = [2 * np.pi * self.alpha ** 3 * (
                    2 * (mu - 1) / (mu + 2))] * 3  # converting N0 from float to 3x3 multiple of identity
            N0_exact = np.diag(N0_exact)
            N0_exact_array += [N0_exact]

        self.N0_snapshots = np.asarray(N0_array).real

        return np.asarray(N0_array).real, np.asarray(N0_exact_array)

    def calc_N0_ROM(self):
        """
        James Elgy - 2022
        Function to calculate N0 given theta0 ROM for each relative permeability.
        Function also calculates the exact N0 for a sphere for testing purposes.
        :return: N0 and N0_exact.
        """

        N0_array = []
        N0_exact_array = []

        for ind, mu in enumerate(self.permeability_array_ROM):  # Assigning permeability
            # print(f'Calculating N0 - mur = {mu} - ind = {ind}')

            Theta0i = GridFunction(self.theta0_fes)
            Theta0j = GridFunction(self.theta0_fes)
            N0 = np.zeros((3, 3), dtype=complex)

            mur = dict(zip(self.object_list, [1 if mat == 'air' else mu for mat in self.mesh.GetMaterials()]))
            mur_coeff = CoefficientFunction([mur[mat] for mat in self.mesh.GetMaterials()])
            # Theta0Sol = np.tile(D[:,ind],(3,1)).transpose()  # For sphere all directions give same result.
            Theta0Sol = self.theta0_ROM[:, ind, :]

            # Calculate the N0 tensor
            VolConstant = Integrate(1 - mur_coeff ** (-1), self.mesh)
            for i in range(3):
                Theta0i.vec.FV().NumPy()[:] = Theta0Sol[:, i]
                for j in range(i + 1):
                    Theta0j.vec.FV().NumPy()[:] = Theta0Sol[:, j]
                    if i == j:
                        N0[i, j] = (self.alpha ** 3) * (VolConstant + (1 / 4) * (
                            Integrate(mur_coeff ** (-1) * (InnerProduct(curl(Theta0i), curl(Theta0j))), self.mesh)))
                    else:
                        N0[i, j] = (self.alpha ** 3 / 4) * (
                            Integrate(mur_coeff ** (-1) * (InnerProduct(curl(Theta0i), curl(Theta0j))), self.mesh))

            # Copy the tensor
            N0 += np.transpose(N0-np.diag(np.diag(N0)))
            N0_array += [N0]

            # Calc exact N0 for sphere
            N0_exact = [2 * np.pi * self.alpha ** 3 * (
                    2 * (mu - 1) / (mu + 2))] * 3  # converting N0 from float to 3x3 multiple of identity
            N0_exact = np.diag(N0_exact)
            N0_exact_array += [N0_exact]

        self.N0_ROM= np.asarray(N0_array).real

        return N0_array, N0_exact_array

    def calc_R_snapshots(self, use_integral=False, use_parallel=False):
        """
        James Elgy - 2022.
        Function to calculate real component of MPT given object parameters alpha, sigma, mur and omega. Function
        requires that theta0 and theta1 snapshots be calculated first.
        Function has the option to calculate R using the integral form used in MPT-calculator, or
        :param use_integral:
        :return:
        """
        if use_integral == True:
            fes = self.theta0_fes
            fes2 = self.theta1_fes
            real_part_MPT = np.zeros((3, 3, len(self.permeability_array), len(self.frequency_array)))
            xivec = [CoefficientFunction((0, -z, y)), CoefficientFunction((z, 0, -x)), CoefficientFunction((-y, x, 0))]

            for ind, mur in enumerate(self.permeability_array):
                for jnd, omega in enumerate(self.frequency_array):

                    mu = {'air': 1.0, 'sphere': mur}
                    mur_coeff = CoefficientFunction([mu[mat] for mat in self.mesh.GetMaterials()])
                    nu = omega * 4 * np.pi * 1e-7 * (self.alpha ** 2)

                    R = np.zeros([3, 3])
                    I = np.zeros([3, 3])
                    Theta0i = GridFunction(fes)
                    Theta0j = GridFunction(fes)
                    Theta1i = GridFunction(fes2)
                    Theta1j = GridFunction(fes2)
                    for i in range(3):
                        Theta0i.vec.FV().NumPy()[:] = self.theta0_snapshots[:, ind, i]
                        xii = xivec[i]
                        if i == 0:
                            Theta1i.vec.FV().NumPy()[:] = self.theta1_snapshots[:,i, ind, jnd]
                        if i == 1:
                            Theta1i.vec.FV().NumPy()[:] = self.theta1_snapshots[:,i, ind, jnd]
                        if i == 2:
                            Theta1i.vec.FV().NumPy()[:] = self.theta1_snapshots[:,i, ind, jnd]
                        for j in range(i + 1):
                            Theta0j.vec.FV().NumPy()[:] = self.theta0_snapshots[:, ind, j]
                            xij = xivec[j]
                            if j == 0:
                                Theta1j.vec.FV().NumPy()[:] = self.theta1_snapshots[:,j, ind, jnd]
                            if j == 1:
                                Theta1j.vec.FV().NumPy()[:] = self.theta1_snapshots[:,j, ind, jnd]
                            if j == 2:
                                Theta1j.vec.FV().NumPy()[:] = self.theta1_snapshots[:,j, ind, jnd]

                            # Real and Imaginary parts
                            R[i, j] = -(((self.alpha ** 3) / 4) * Integrate(
                                (mur_coeff ** (-1)) * (curl(Theta1j) * Conj(curl(Theta1i))), self.mesh)).real
                    R += np.transpose(R - np.diag(np.diag(R))).real
                    real_part_MPT[:,:, ind, jnd] = R
        else:
            u, v = self.theta1_fes.TnT()

            K2 = BilinearForm(self.theta1_fes, symmetric=True)
            K2 += SymbolicBFI(self.inout * curl(u) * curl(v))
            K2.Assemble()

            K1 = BilinearForm(self.theta1_fes, symmetric=True)
            K1 += SymbolicBFI((1 - self.inout) * curl(u) * curl(v))
            K1.Assemble()

            rows, cols, vals = (K1.mat).COO()
            A1 = sp.csr_matrix((vals, (rows, cols)))
            rows, cols, vals = (K2.mat).COO()
            A2 = sp.csr_matrix((vals, (rows, cols)))

            if use_parallel == False:
                real_part_MPT = np.zeros((3, 3, len(self.permeability_array), len(self.frequency_array)))
                count = 1
                for i in range(3):
                    for j in range(3):
                        for ind, mur in enumerate(self.permeability_array):
                            for jnd, omega in enumerate(self.frequency_array):
                                print(f'Calculating R for omega={omega:.2f}, mur={mur:.2f}: {count}/{9*len(self.permeability_array)*len(self.frequency_array)}',end='\r' )
                                A = A1 + (A2 / mur)
                                R = self.theta1_snapshots[:, i, ind, jnd][None, :] @ A @ np.conj(self.theta1_snapshots[:, j, ind, jnd])[:, None]
                                R *= (-(self.alpha ** 3) / 4)
                                real_part_MPT[i,j,ind, jnd] = R.real
                                count += 1

            else:
                real_part_MPT = self._calc_R_parallel(A1, A2)

        self.R = real_part_MPT

    def calc_R_ROM(self, use_integral=False):
        """
        James Elgy - 2022.
        Function to calculate real component of MPT given object parameters alpha, sigma, mur and omega. Function
        requires that theta0 and theta1 ROMS be calculated first.
        :return:
        """

        if use_integral == True:
            fes = self.theta0_fes
            fes2 = self.theta1_fes
            real_part_MPT = np.zeros((3, 3, len(self.permeability_array_ROM), len(self.frequency_array_ROM)))
            xivec = [CoefficientFunction((0, -z, y)), CoefficientFunction((z, 0, -x)), CoefficientFunction((-y, x, 0))]

            for ind, mur in enumerate(self.permeability_array_ROM):
                for jnd, omega in enumerate(self.frequency_array_ROM):

                    # mu = {'air': 1.0, 'sphere': mur}
                    mu = dict(
                        zip(self.object_list, [1 if mat == 'air' else mur for mat in self.mesh.GetMaterials()]))
                    mur_coeff = CoefficientFunction([mu[mat] for mat in self.mesh.GetMaterials()])
                    nu = omega * 4 * np.pi * 1e-7 * (self.alpha ** 2)

                    R = np.zeros([3, 3])
                    I = np.zeros([3, 3])
                    Theta0i = GridFunction(fes)
                    Theta0j = GridFunction(fes)
                    Theta1i = GridFunction(fes2)
                    Theta1j = GridFunction(fes2)
                    for i in range(3):
                        Theta0i.vec.FV().NumPy()[:] = self.theta0_ROM[:, ind, i]
                        xii = xivec[i]
                        if i == 0:
                            Theta1i.vec.FV().NumPy()[:] = self.theta1_ROM[:, i, ind, jnd]
                        if i == 1:
                            Theta1i.vec.FV().NumPy()[:] = self.theta1_ROM[:, i, ind, jnd]
                        if i == 2:
                            Theta1i.vec.FV().NumPy()[:] = self.theta1_ROM[:, i, ind, jnd]
                        for j in range(i + 1):
                            Theta0j.vec.FV().NumPy()[:] = self.theta0_ROM[:, ind, j]
                            xij = xivec[j]
                            if j == 0:
                                Theta1j.vec.FV().NumPy()[:] = self.theta1_ROM[:, j, ind, jnd]
                            if j == 1:
                                Theta1j.vec.FV().NumPy()[:] = self.theta1_ROM[:, j, ind, jnd]
                            if j == 2:
                                Theta1j.vec.FV().NumPy()[:] = self.theta1_ROM[:, j, ind, jnd]

                            # Real and Imaginary parts
                            R[i, j] = -(((self.alpha ** 3) / 4) * Integrate(
                                (mur_coeff ** (-1)) * (curl(Theta1j) * Conj(curl(Theta1i))), self.mesh)).real
                    R += np.transpose(R - np.diag(np.diag(R))).real
                    real_part_MPT[:, :, ind, jnd] = R
        else:
            u, v = self.theta1_fes.TnT()

            K2 = BilinearForm(self.theta1_fes, symmetric=True)
            K2 += SymbolicBFI(self.inout * curl(u) * curl(v))
            K2.Assemble()

            K1 = BilinearForm(self.theta1_fes, symmetric=True)
            K1 += SymbolicBFI((1 - self.inout) * curl(u) * curl(v))
            K1.Assemble()

            rows, cols, vals = (K1.mat).COO()
            A1 = sp.csr_matrix((vals, (rows, cols)))
            rows, cols, vals = (K2.mat).COO()
            A2 = sp.csr_matrix((vals, (rows, cols)))

            real_part_MPT = np.zeros((3, 3, len(self.permeability_array_ROM), len(self.frequency_array_ROM)))
            count = 1
            for i in range(3):
                    for j in range(3):
                        for ind, mur in enumerate(self.permeability_array_ROM):
                            for jnd, omega in enumerate(self.frequency_array_ROM):
                                print(f'Calculating R for omega={omega:.2f}, mur={mur:.2f}: {count}/{9*len(self.permeability_array_ROM)*len(self.frequency_array_ROM)}', end='\r')
                                A = A1 + (A2 / mur)
                                R = self.theta1_ROM[:, i, ind, jnd][None, :] @ A @ np.conj(self.theta1_ROM[:, j, ind, jnd])[:, None]
                                R *= (-(self.alpha ** 3) / 4)
                                real_part_MPT[i,j,ind, jnd] = R.real
                                count += 1


        self.R_ROM = real_part_MPT

    def calc_I_snapshots(self, use_integral=False):
        """
        James Elgy - 2022.
        Function to calculate imag component of MPT given object parameters alpha, sigma, mur and omega. Function
        requires that theta0 and theta1 snapshots be calculated first.
        :return:
        """
        # print('Calculating Imaginary Component')

        # Redefininig theta0_fes to be complex. This is so that we can form a compound space
        # dom_nrs_metal = [0 if mat == "air" else 1 for mat in self.mesh.GetMaterials()]
        # theta0_fes = HCurl(self.mesh, order=self.order, dirichlet='outer', complex=True, flags={'nograds': True})
        # theta1_fes = HCurl(self.mesh, order=self.order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)

        # Building matrices:
        # joint_fes = theta0_fes*theta1_fes
        # c, u = joint_fes.TrialFunction()
        # w, v = joint_fes.TestFunction()

        if use_integral == True:
            imag_part_MPT = np.zeros((3,3,len(self.permeability_array), len(self.frequency_array)))

            fes = self.theta0_fes
            fes2 = self.theta1_fes
            real_part_MPT = np.zeros((3, 3, len(self.permeability_array), len(self.frequency_array)))
            xivec = [CoefficientFunction((0, -z, y)), CoefficientFunction((z, 0, -x)), CoefficientFunction((-y, x, 0))]
            sig_coeff = CoefficientFunction([self.sigma_dict[mat] for mat in self.mesh.GetMaterials()])
            for ind, mur in enumerate(self.permeability_array):
                for jnd, omega in enumerate(self.frequency_array):

                    mu = dict(
                        zip(self.object_list, [1 if mat == 'air' else mur for mat in self.mesh.GetMaterials()]))
                    mur_coeff = CoefficientFunction([mu[mat] for mat in self.mesh.GetMaterials()])
                    nu = omega * 4 * np.pi * 1e-7 * (self.alpha ** 2)

                    R = np.zeros([3, 3])
                    I = np.zeros([3, 3])
                    Theta0i = GridFunction(fes)
                    Theta0j = GridFunction(fes)
                    Theta1i = GridFunction(fes2)
                    Theta1j = GridFunction(fes2)
                    for i in range(3):
                        Theta0i.vec.FV().NumPy()[:] = self.theta0_snapshots[:, ind, i]
                        xii = xivec[i]
                        if i == 0:
                            Theta1i.vec.FV().NumPy()[:] = self.theta1_snapshots[:,i, ind, jnd]
                        if i == 1:
                            Theta1i.vec.FV().NumPy()[:] = self.theta1_snapshots[:,i, ind, jnd]
                        if i == 2:
                            Theta1i.vec.FV().NumPy()[:] = self.theta1_snapshots[:,i, ind, jnd]
                        for j in range(i + 1):
                            Theta0j.vec.FV().NumPy()[:] = self.theta0_snapshots[:, ind, j]
                            xij = xivec[j]
                            if j == 0:
                                Theta1j.vec.FV().NumPy()[:] = self.theta1_snapshots[:,j, ind, jnd]
                            if j == 1:
                                Theta1j.vec.FV().NumPy()[:] = self.theta1_snapshots[:,j, ind, jnd]
                            if j == 2:
                                Theta1j.vec.FV().NumPy()[:] = self.theta1_snapshots[:,j, ind, jnd]

                            # Real and Imaginary parts
                            # R[i, j] = -(((self.alpha ** 3) / 4) * Integrate(
                            #     (mur_coeff ** (-1)) * (curl(Theta1j) * Conj(curl(Theta1i))), self.mesh)).real
                            I[i, j] = ((self.alpha**3)/4)*Integrate(self.inout*nu*sig_coeff*((Theta1j+Theta0j+xij)*(Conj(Theta1i)+Theta0i+xii)),self.mesh).real
                    I += np.transpose(I - np.diag(np.diag(I))).real
                    imag_part_MPT[:,:, ind, jnd] = I
        else:
            sigma_coeff = CoefficientFunction([self.sigma_dict[mat] for mat in self.mesh.GetMaterials()])
            nu_no_omega = (4*np.pi*1e-7) * (self.alpha**2) * sigma_coeff

            u,v = self.theta1_fes.TnT()

            A = BilinearForm(self.theta1_fes, symmetric=True)
            A += SymbolicBFI(nu_no_omega*self.inout*(v*u))
            A.Assemble()

            # B = BilinearForm(joint_fes)
            # B += SymbolicBFI(self.inout*(c*v))
            # B.Assemble()
            #
            # C = BilinearForm(joint_fes)
            # C += SymbolicBFI(self.inout * (u*w))
            # C.Assemble()
            #
            # D = BilinearForm(joint_fes, symmetric=True)
            # D += SymbolicBFI(self.inout * (u*v))
            # D.Assemble()

            # Converting to scipy sparce matrices:
            rows, cols, vals = A.mat.COO()
            A_mat = ((self.alpha**3) / 4) * sp.csr_matrix((vals, (rows, cols)))

            # rows, cols, vals = B.mat.COO()
            # B_mat = sp.csr_matrix((vals, (rows, cols)))
            #
            # rows, cols, vals = C.mat.COO()
            # C_mat = sp.csr_matrix((vals, (rows, cols)))
            #
            # rows, cols, vals = D.mat.COO()
            # D_mat = sp.csr_matrix((vals, (rows, cols)))

            # Preallocating
            E = np.zeros((3,self.theta1_fes.ndof), dtype=complex)
            # F = np.zeros((3,joint_fes.ndof), dtype=complex)
            G = np.zeros((3,3))

            for i in range(3):

                E_lf = LinearForm(self.theta1_fes)
                E_lf += SymbolicLFI(nu_no_omega * self.inout * self.xivec[i] * v)
                E_lf.Assemble()
                E[i,:] = ((self.alpha**3) / 4) * E_lf.vec.FV().NumPy()[:]

                #
                # F_lf = LinearForm(joint_fes)
                # F_lf += SymbolicLFI(self.inout * self.xivec[i] * v)
                # F_lf.Assemble()
                # F[i, :] = F_lf.vec.FV().NumPy()[:]

                for j in range(3):
                    G[i,j] = ((self.alpha**3) / 4) * Integrate(nu_no_omega * self.inout * self.xivec[i] * self.xivec[j], self.mesh)

            # E = sp.csr_array(E)
            # F = sp.csr_array(F)
            H = E.transpose()
            # J = F.transpose()

            imag_part_MPT = np.zeros((3,3,len(self.permeability_array), len(self.frequency_array)))
            count = 1
            for i in range(3):
                for j in range(3):
                    for ind, mur in enumerate(self.permeability_array):
                        for jnd, omega in enumerate(self.frequency_array):
                            print(f'Calculating I for omega={omega:.2f}, mur={mur:.2f}: {count}/{9*len(self.permeability_array)*len(self.frequency_array)}',end='\r')
                            total = 0
                            t0 = np.squeeze(self.theta0_snapshots[:, ind, :]) + 1j*np.zeros(self.theta0_snapshots[:, ind, :].shape)
                            t1 = np.squeeze(self.theta1_snapshots[:, :, ind, jnd])

                            c1 = (t0[:, i])[None, :] @ (A_mat*omega) @ (t0[:, j])[:, None]
                            c2 = (t1[:, i])[None, :] @ (A_mat*omega) @ (t0[:, j])[:, None]
                            c3 = (t0[:, i])[None, :] @ (A_mat*omega) @ np.conj(t1[:, j])[:, None]
                            c4 = (t1[:, i])[None, :] @ (A_mat*omega) @ np.conj(t1[:, j])[:, None]

                            c5 = (E[i, :]*omega) @ t0[:, j][:, None]
                            c6 = (E[i, :]*omega) @ np.conj(t1[:, j])[:, None]
                            c7 = G[i, j]*omega
                            c8 = (t0[:, i][None,:] @ ((H[:,j]*omega)))
                            c9 = (t1[:, i][None,:] @ ((H[:,j]*omega)))

                            total = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9
                            imag_part_MPT[i, j, ind, jnd] = complex(total).real
                            count += 1

        self.I = imag_part_MPT.real
        return imag_part_MPT.real

    def calc_I_ROM(self, use_integral=False):
        """
                James Elgy - 2022.
                Function to calculate imag component of MPT given object parameters alpha, sigma, mur and omega. Function
                requires that theta0 and theta1 snapshots be calculated first.
                :return:
                """
        # print('Calculating Imaginary Component')

        # Redefininig theta0_fes to be complex. This is so that we can form a compound space
        # dom_nrs_metal = [0 if mat == "air" else 1 for mat in self.mesh.GetMaterials()]
        # theta0_fes = HCurl(self.mesh, order=self.order, dirichlet='outer', complex=True, flags={'nograds': True})
        # theta1_fes = HCurl(self.mesh, order=self.order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)

        # Building matrices:
        # joint_fes = theta0_fes*theta1_fes
        # c, u = joint_fes.TrialFunction()
        # w, v = joint_fes.TestFunction()
        if use_integral == True:
            imag_part_MPT = np.zeros((3,3,len(self.permeability_array_ROM), len(self.frequency_array_ROM)))

            fes = self.theta0_fes
            fes2 = self.theta1_fes
            real_part_MPT = np.zeros((3, 3, len(self.permeability_array), len(self.frequency_array)))
            xivec = [CoefficientFunction((0, -z, y)), CoefficientFunction((z, 0, -x)), CoefficientFunction((-y, x, 0))]
            sig_coeff = CoefficientFunction([self.sigma_dict[mat] for mat in self.mesh.GetMaterials()])
            for ind, mur in enumerate(self.permeability_array_ROM):
                for jnd, omega in enumerate(self.frequency_array_ROM):

                    mu = dict(
                        zip(self.object_list, [1 if mat == 'air' else mur for mat in self.mesh.GetMaterials()]))
                    mur_coeff = CoefficientFunction([mu[mat] for mat in self.mesh.GetMaterials()])
                    nu = omega * 4 * np.pi * 1e-7 * (self.alpha ** 2)

                    R = np.zeros([3, 3])
                    I = np.zeros([3, 3])
                    Theta0i = GridFunction(fes)
                    Theta0j = GridFunction(fes)
                    Theta1i = GridFunction(fes2)
                    Theta1j = GridFunction(fes2)
                    for i in range(3):
                        Theta0i.vec.FV().NumPy()[:] = self.theta0_ROM[:, ind, i]
                        xii = xivec[i]
                        if i == 0:
                            Theta1i.vec.FV().NumPy()[:] = self.theta1_ROM[:,i, ind, jnd]
                        if i == 1:
                            Theta1i.vec.FV().NumPy()[:] = self.theta1_ROM[:,i, ind, jnd]
                        if i == 2:
                            Theta1i.vec.FV().NumPy()[:] = self.theta1_ROM[:,i, ind, jnd]
                        for j in range(i + 1):
                            Theta0j.vec.FV().NumPy()[:] = self.theta0_ROM[:, ind, j]
                            xij = xivec[j]
                            if j == 0:
                                Theta1j.vec.FV().NumPy()[:] = self.theta1_ROM[:,j, ind, jnd]
                            if j == 1:
                                Theta1j.vec.FV().NumPy()[:] = self.theta1_ROM[:,j, ind, jnd]
                            if j == 2:
                                Theta1j.vec.FV().NumPy()[:] = self.theta1_ROM[:,j, ind, jnd]

                            # Real and Imaginary parts
                            # R[i, j] = -(((self.alpha ** 3) / 4) * Integrate(
                            #     (mur_coeff ** (-1)) * (curl(Theta1j) * Conj(curl(Theta1i))), self.mesh)).real
                            I[i, j] = ((self.alpha**3)/4)*Integrate(self.inout*nu*sig_coeff*((Theta1j+Theta0j+xij)*(Conj(Theta1i)+Theta0i+xii)),self.mesh).real
                    I += np.transpose(I - np.diag(np.diag(I))).real
                    imag_part_MPT[:,:, ind, jnd] = I
        else:
            sigma_coeff = CoefficientFunction([self.sigma_dict[mat] for mat in self.mesh.GetMaterials()])
            nu_no_omega = 4 * np.pi * 1e-7 * self.alpha ** 2 * sigma_coeff

            u, v = self.theta1_fes.TnT()

            A = BilinearForm(self.theta1_fes, symmetric=True)
            A += SymbolicBFI(nu_no_omega * self.inout * (u * v))
            A.Assemble()

            # B = BilinearForm(joint_fes)
            # B += SymbolicBFI(self.inout*(c*v))
            # B.Assemble()
            #
            # C = BilinearForm(joint_fes)
            # C += SymbolicBFI(self.inout * (u*w))
            # C.Assemble()
            #
            # D = BilinearForm(joint_fes, symmetric=True)
            # D += SymbolicBFI(self.inout * (u*v))
            # D.Assemble()

            # Converting to scipy sparce matrices:
            rows, cols, vals = A.mat.COO()
            A_mat = sp.csr_matrix((vals, (rows, cols)))

            # rows, cols, vals = B.mat.COO()
            # B_mat = sp.csr_matrix((vals, (rows, cols)))
            #
            # rows, cols, vals = C.mat.COO()
            # C_mat = sp.csr_matrix((vals, (rows, cols)))
            #
            # rows, cols, vals = D.mat.COO()
            # D_mat = sp.csr_matrix((vals, (rows, cols)))

            # Preallocating
            E = np.zeros((3, self.theta1_fes.ndof), dtype=complex)
            # F = np.zeros((3,joint_fes.ndof), dtype=complex)
            G = np.zeros((3, 3))

            for i in range(3):

                E_lf = LinearForm(self.theta1_fes)
                E_lf += SymbolicLFI(nu_no_omega * self.inout * self.xivec[i] * u)
                E_lf.Assemble()
                E[i, :] = E_lf.vec.FV().NumPy()[:]

                #
                # F_lf = LinearForm(joint_fes)
                # F_lf += SymbolicLFI(self.inout * self.xivec[i] * v)
                # F_lf.Assemble()
                # F[i, :] = F_lf.vec.FV().NumPy()[:]

                for j in range(3):
                    G[i, j] = Integrate(nu_no_omega * self.inout * self.xivec[i] * self.xivec[j], self.mesh)

            # E = sp.csr_array(E)
            # F = sp.csr_array(F)
            H = E.transpose()
            # J = F.transpose()

            imag_part_MPT = np.zeros((3, 3, len(self.permeability_array_ROM), len(self.frequency_array_ROM)))
            count = 1
            for i in range(3):
                for j in range(3):
                    for ind, mur in enumerate(self.permeability_array_ROM):
                        for jnd, omega in enumerate(self.frequency_array_ROM):
                            print(f'Calculating I for omega={omega:.2f}, mur={mur:.2f} {count}/{9*len(self.permeability_array_ROM)*len(self.frequency_array_ROM)}', end='\r')
                            total = 0
                            t0 = np.squeeze(self.theta0_ROM[:, ind, :]) + 1j*np.zeros(self.theta0_ROM[:, ind, :].shape)
                            t1 = np.squeeze(self.theta1_ROM[:, :, ind, jnd])

                            c1 = (t0[:, i])[None, :] @ A_mat @ (t0[:, j])[:, None]
                            c2 = (t1[:, i])[None, :] @ A_mat @ (t0[:, j])[:, None]
                            c3 = (t0[:, i])[None, :] @ A_mat @ (t1[:, j])[:, None]
                            c4 = (t1[:, i])[None, :] @ A_mat @ np.conj(t1[:, j])[:, None]

                            c5 = E[i, :] @ t0[:, j]
                            c6 = E[i, :] @ np.conj(t1[:, j])
                            c7 = G[i, j]
                            c8 = (t0[:, i][None, :] @ (E.transpose()))[:, j]
                            c9 = (t1[:, i][None, :] @ (E.transpose()))[:, j]

                            total = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9
                            imag_part_MPT[i, j, ind, jnd] = (self.alpha ** 3 / 4) * complex(total).real * omega
                            count += 1

        self.I_ROM = imag_part_MPT.real
        return imag_part_MPT.real

    def calc_eddy_current_limit(self):
        """
        James Elgy - 2022:
        function to calculate when the wavelength is smaller than the object size. This is then returned as an
        upper limit on the eddy current approximation.
        :return:
        """

        # c = 299792458  # Speed of light m/s

        limit_frequency = []
        for mur in self.permeability_array:
            epsilon = 8.854e-12
            mu = mur * 4 * np.pi * 1e-7
            k = np.sqrt(self.frequency_array**2 * epsilon * mu + 1j * mu * self.sigma_dict['sphere'] * self.frequency_array)
            wavelength = 2*np.pi/k.real
            fudge_factor = 120 # alpha uses a global scaling but the geo file may not define a unit object.
            # wavelength = c / (np.sqrt(1*mur) * (self.frequency_array/(2*np.pi)))
            for lam, freq in zip(wavelength, self.frequency_array):
                if lam <= self.alpha*fudge_factor:
                    max_frequency = freq
                    break
                else:
                    max_frequency = np.nan
            limit_frequency += [max_frequency]

        return limit_frequency

    def calc_error_bounds(self):
        pass

    def calc_single_theta0(self, mur, model_array='default', Sigma='default', U_trunc='default'):

        if model_array == 'default':
            model_array = self.theta0_NN_models
        if U_trunc == 'default':
            U_trunc = self.theta0_U_truncated_array
        if Sigma == 'default':
            S_trunc = self.theta0_Sigma_truncated_array

        theta0 = np.zeros((len(U_trunc[0]),3))

        for dim in range(3):
            Vh_pred = np.zeros((len(S_trunc[dim]), 1))
            model_dim = model_array[dim]
            for mode in range(len(S_trunc[dim])):
                model = model_dim[mode][0]
                Vh_pred[mode,:] = self._make_prediction_1D(mur, model)
            theta0[:, dim] = np.squeeze(np.asarray(U_trunc[dim]) @ np.asarray(S_trunc[dim]) @ Vh_pred)

        self.theta0_ROM = theta0[:,None,:]
        return theta0

    def calc_single_theta1(self, mur, omega, model_array='default', Sigma='default', U_trunc='default'):

        if model_array == 'default':
            model_array = self.theta1_NN_models
        if U_trunc == 'default':
            U_trunc = self.theta1_U_truncated_array
        if Sigma == 'default':
            S_trunc = self.theta1_Sigma_truncated_array

        theta1 = np.zeros((len(U_trunc[0]),3), dtype=complex)
        # Vh_pred = np.zeros((len(S_trunc[0]), 3), dtype=complex)
        for dim in range(3):

            Vh_pred = np.zeros((len(S_trunc[dim]),1), dtype=complex)
            Vh_pred_real = np.zeros((len(S_trunc[dim]), 1))
            Vh_pred_imag = np.zeros((len(S_trunc[dim]), 1))

            model_dim = model_array[dim]
            for mode in range(len(S_trunc[dim])):
                model_real = model_dim[mode][0]
                model_imag = model_dim[mode][1]
                Vh_pred_real[mode,:] = self._make_prediction_2D(mur, omega, model_real)
                Vh_pred_imag[mode,:] = self._make_prediction_2D(mur, omega, model_imag)
                Vh_pred[mode, :] = Vh_pred_real[mode, :] + 1j * Vh_pred_imag[mode, :]
            theta1[:, dim] = np.squeeze(np.asarray(U_trunc[dim]) @ np.asarray(S_trunc[dim]) @ Vh_pred)

        self.theta1_ROM = theta1[:,:,None, None]
        return theta1

    def plot_N0(self, plot_all=False, use_ROM=False):
        """
        James Elgy - 2022
        Function to plot N0 components.
        :param plot_all: bool: Option to plot all components of N0.
        :param use_ROM: bool: Option to use ROM values of N0. Otherwise uses snapshot values.
        :return:
        """
        if use_ROM == True:
            N0 = self.N0_ROM
            permeability_array = self.permeability_array_ROM
        else:
            N0 = self.N0_snapshots
            permeability_array = self.permeability_array

        plt.figure()

        plt.semilogx(permeability_array, N0[:,0,0], label='$\mathcal{N}^0_{0,0}$')
        plt.semilogx(permeability_array, N0[:,1,1], label='$\mathcal{N}^0_{1,1}$')
        plt.semilogx(permeability_array, N0[:,2,2], label='$\mathcal{N}^0_{2,2}$')
        if plot_all == True:
            plt.semilogx(permeability_array, N0[:, 0, 1], label='$\mathcal{N}^0_{0,1}$')
            plt.semilogx(permeability_array, N0[:, 0, 2], label='$\mathcal{N}^0_{0,2}$')
            plt.semilogx(permeability_array, N0[:, 1, 2], label='$\mathcal{N}^0_{1,2}$')
        plt.legend()
        plt.xlabel('$\mu_r$')
        plt.ylabel('$\mathcal{N}^0$')
        plt.title('$\mathcal{N}^0$')
        if use_ROM is True:
            plt.title('$\mathcal{N}^0$ ROM')

    def plot_real_component(self, style='line', plot_all=False, use_ROM=False, add_eddy_current_limit=False, plot_errorbars=True):
        """
        James Elgy - 2022
        Function to plot real components.
        :param plot_all: bool: Option to plot all components of N0.
        :param style: 'line' Flag for plotting style. Line produces a series of line plots. 'surface' plots as a
        surface.
        :param use_ROM: bool: Option to use ROM values of N0. Otherwise uses snapshot values.
        :return:
        """

        plt.figure()
        if style == 'line':
            #  define colour map
            if use_ROM == False:
                max_c_value = max(np.log10(self.permeability_array))
                min_c_value = min(np.log10(self.permeability_array))
                R = self.R
                N0 = self.N0_snapshots
                frequency_array = self.frequency_array
                permeability_array = self.permeability_array
            else:
                max_c_value = max(np.log10(self.permeability_array_ROM))
                min_c_value = min(np.log10(self.permeability_array_ROM))
                R = self.R_ROM
                N0 = self.N0_ROM
                frequency_array = self.frequency_array_ROM
                permeability_array = self.permeability_array_ROM

            normalised_permeability_array = (np.log10(permeability_array) - min_c_value) / \
                                            (max_c_value - min_c_value)
            cmap = plt.cm.get_cmap('jet')

            for ind, nmur in enumerate(normalised_permeability_array):
                rgba = cmap(nmur)
                plt.semilogx(frequency_array, R[0,0,ind,:] + N0[ind,0,0], color=rgba)
            sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=min_c_value,
                                                                      vmax=max_c_value))
            cbar = plt.colorbar(sm)
            cbar.ax.locator_params(nbins=4)
            cbar.set_label('$\mathrm{log}_{10}(\mu_r)$', rotation=270, labelpad=15)
            plt.xlabel('$\omega$, rad/s')
            plt.ylabel('$(\mathcal{R} + \mathcal{N}^0)_{0,0}$')
            plt.title('$\mathcal{R} + \mathcal{N}^0$')
            if use_ROM is True:
                plt.title('$\mathcal{R} + \mathcal{N}^0$ ROM')

            if plot_all == True:
                plt.figure()
                for ind, nmur in enumerate(normalised_permeability_array):
                    rgba = cmap(nmur)
                    plt.semilogx(frequency_array, R[1, 1, ind, :] + N0[ind, 1, 1], color=rgba)
                cbar = plt.colorbar(sm)
                cbar.ax.locator_params(nbins=4)
                cbar.set_label('$\mathrm{log}_{10}(\mu_r)$', rotation=270, labelpad=15)
                plt.xlabel('$\omega$, rad/s')
                plt.ylabel('$(\mathcal{R} + \mathcal{N}^0)_{1,1}$')
                plt.title('$\mathcal{R} + \mathcal{N}^0$')
                if use_ROM is True:
                    plt.title('$\mathcal{R} + \mathcal{N}^0$ ROM')

                plt.figure()
                for ind, nmur in enumerate(normalised_permeability_array):
                    rgba = cmap(nmur)
                    plt.semilogx(frequency_array, R[2, 2, ind, :] + N0[ind, 2, 2], color=rgba)
                cbar = plt.colorbar(sm)
                cbar.ax.locator_params(nbins=4)
                cbar.set_label('$\mathrm{log}_{10}(\mu_r)$', rotation=270, labelpad=15)
                plt.xlabel('$\omega$, rad/s')
                plt.ylabel('$(\mathcal{R} + \mathcal{N}^0)_{2,2}$')
                plt.title('$\mathcal{R} + \mathcal{N}^0$')
                if use_ROM is True:
                    plt.title('$\mathcal{R} + \mathcal{N}^0$ ROM')

                plt.figure()
                for ind, nmur in enumerate(normalised_permeability_array):
                    rgba = cmap(nmur)
                    plt.semilogx(frequency_array, R[0, 1, ind, :] + N0[ind, 0, 1], color=rgba)
                cbar = plt.colorbar(sm)
                cbar.ax.locator_params(nbins=4)
                cbar.set_label('$\mathrm{log}_{10}(\mu_r)$', rotation=270, labelpad=15)
                plt.xlabel('$\omega$, rad/s')
                plt.ylabel('$(\mathcal{R} + \mathcal{N}^0)_{0,1}$')
                plt.title('$\mathcal{R} + \mathcal{N}^0$')
                if use_ROM is True:
                    plt.title('$\mathcal{R} + \mathcal{N}^0$ ROM')

                plt.figure()
                for ind, nmur in enumerate(normalised_permeability_array):
                    rgba = cmap(nmur)
                    plt.semilogx(frequency_array, R[0, 2, ind, :] + N0[ind, 0, 2], color=rgba)
                cbar = plt.colorbar(sm)
                cbar.ax.locator_params(nbins=4)
                cbar.set_label('$\mathrm{log}_{10}(\mu_r)$', rotation=270, labelpad=15)
                plt.xlabel('$\omega$, rad/s')
                plt.ylabel('$(\mathcal{R} + \mathcal{N}^0)_{0,2}$')
                plt.title('$\mathcal{R} + \mathcal{N}^0$')
                if use_ROM is True:
                    plt.title('$\mathcal{R} + \mathcal{N}^0$ ROM')

                plt.figure()
                for ind, nmur in enumerate(normalised_permeability_array):
                    rgba = cmap(nmur)
                    plt.semilogx(frequency_array, R[1, 2, ind, :] + N0[ind, 1, 2], color=rgba)
                cbar = plt.colorbar(sm)
                cbar.ax.locator_params(nbins=4)
                cbar.set_label('$\mathrm{log}_{10}(\mu_r)$', rotation=270, labelpad=15)
                plt.xlabel('$\omega$, rad/s')
                plt.ylabel('$(\mathcal{R} + \mathcal{N}^0)_{1,2}$')
                plt.title('$\mathcal{R} + \mathcal{N}^0$')
                if use_ROM is True:
                    plt.title('$\mathcal{R} + \mathcal{N}^0$ ROM')

        else:

            # 3d matplotlib figures do not work well with a log scale. Thus we introduce this tick formatter.
            import matplotlib.ticker as mticker
            def log_tick_formatter(val, pos=None):
                if val == 0:
                    return '$10^0$'
                else:
                    return '$10^{' + f'{val:.0f}' + '}$'

            if use_ROM == False:
                R = self.R
                N0 = self.N0_snapshots
                frequency_array = self.frequency_array
                permeability_array = self.permeability_array
            else:
                R = self.R_ROM
                N0 = self.N0_ROM
                frequency_array = self.frequency_array_ROM
                permeability_array = self.permeability_array_ROM

            xx,yy = np.meshgrid(np.log10(frequency_array), np.log10(permeability_array))
            zz = np.ones(xx.shape)

            for ind in range(len(permeability_array)):
                zz[ind,:] = R[0, 0, ind, :] + N0[ind, 0, 0]
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            if use_ROM is False:
                ax.plot_wireframe(xx,yy,zz)
            else:
                ax.plot_wireframe(xx, yy, zz, label='ML prediction')
            ax.set_xlabel('$\omega$ [rad/s]')
            ax.set_ylabel('$\mu_r$')
            ax.set_zlabel('$(\mathcal{R} + \mathcal{N}^0)_{0,0}$ [m$^3$]')

            # Here, we are setting the axis tick labels to be formatted as 10^n for neater presentation.
            startx, endx = ax.get_xlim()
            starty, endy = ax.get_ylim()
            ax.xaxis.set_ticks(np.round(np.arange(startx, endx, 1)))
            ax.yaxis.set_ticks(np.round(np.arange(starty, endy, 1)))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))

            # Option to add snapshot solutions to ROM surface plot.
            overlay_snapshots = True
            if (use_ROM is True) and (overlay_snapshots is True):
                xx_snapshots, yy_snapshots = np.meshgrid(np.log10(self.frequency_array), np.log10(self.permeability_array))
                zz = np.zeros(xx_snapshots.shape)

                for ind in range(len(self.permeability_array)):
                    zz[ind, :] = self.R[0, 0, ind, :] + self.N0_snapshots[ind, 0, 0]
                ax.scatter(np.ravel(xx_snapshots),np.ravel(yy_snapshots),np.ravel(zz),
                           color='r',
                           label='Snapshot Solutions')

            # Option to plot intersecting surface at the eddy current limit where wavelength <= object size.
            # add_eddy_current_limit = True
            if add_eddy_current_limit is True:
                max_omega = self.calc_eddy_current_limit()
                z_max = ax.get_zlim()[1]
                z_min = ax.get_zlim()[0]

                yy,zz = np.meshgrid(np.log10(self.permeability_array), np.linspace(z_min, z_max, 50))
                xx = np.ones(yy.shape)

                for ind, omega in enumerate(max_omega):
                    xx[:,ind] = np.log10(omega)

                surf = ax.plot_surface(xx,yy,zz, color='m', alpha=0.4, linewidth=0, label='Eddy current limit')
                # 3d plotting in matplotlib is still buggy. Introduces edgecolors2d and facecolors2d in order to
                # add legend entry.
                surf._facecolors2d = surf._facecolor3d
                surf._edgecolors2d = surf._edgecolor3d

            if plot_errorbars is True:
                # component 1:
                zz = np.zeros(xx.shape)
                for ind in range(len(permeability_array)):
                    zz[ind, :] = R[0, 0, ind, :] + N0[ind, 0, 0]
                    for jnd in range(len(frequency_array)):
                        zz[ind, jnd] += self.error_tensors[ind, jnd, 0]
                ax.plot_wireframe(xx,yy,zz, alpha=0.4,color='g', label='Upper/Lower limit')

                zz = np.zeros(xx.shape)
                for ind in range(len(permeability_array)):
                    zz[ind, :] = R[0, 0, ind, :] + N0[ind, 0, 0]
                    for jnd in range(len(frequency_array)):
                        zz[ind, jnd] -= self.error_tensors[ind, jnd, 0]
                ax.plot_wireframe(xx,yy,zz, alpha=0.4,color='g')

            plt.legend()
            plt.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))

    def plot_imag_component(self, style='line', plot_all=False, use_ROM=False, add_eddy_current_limit=False, plot_errorbars=True):
        """
        James Elgy - 2022
        Function to plot imag components.
        :param plot_all: bool: Option to plot all components of N0.
        :param style: 'line' Flag for plotting style. Line produces a series of line plots. 'surface' plots as a
        surface.
        :param use_ROM: bool: Option to use ROM values of N0. Otherwise uses snapshot values.
        :return:
        """

        plt.figure()
        if style == 'line':
            #  define colour map
            if use_ROM == False:
                max_c_value = max(np.log10(self.permeability_array))
                min_c_value = min(np.log10(self.permeability_array))
                I = self.I
                frequency_array = self.frequency_array
                permeability_array = self.permeability_array
            else:
                max_c_value = max(np.log10(self.permeability_array_ROM))
                min_c_value = min(np.log10(self.permeability_array_ROM))
                I = self.I_ROM
                frequency_array = self.frequency_array_ROM
                permeability_array = self.permeability_array_ROM

            normalised_permeability_array = (np.log10(permeability_array) - min_c_value) / \
                                            (max_c_value - min_c_value)
            cmap = plt.cm.get_cmap('jet')

            for ind, nmur in enumerate(normalised_permeability_array):
                rgba = cmap(nmur)
                plt.semilogx(frequency_array, I[0,0,ind,:], color=rgba)
            sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=min_c_value,
                                                                      vmax=max_c_value))
            cbar = plt.colorbar(sm)
            cbar.ax.locator_params(nbins=4)
            cbar.set_label('$\mathrm{log}_{10}(\mu_r)$', rotation=270, labelpad=15)
            plt.xlabel('$\omega$, rad/s')
            plt.ylabel('$(\mathcal{I})_{0,0}$')
            plt.title('$\mathcal{I}$')
            if use_ROM is True:
                plt.title('$\mathcal{I}$ ROM')

            if plot_all == True:
                plt.figure()
                for ind, nmur in enumerate(normalised_permeability_array):
                    rgba = cmap(nmur)
                    plt.semilogx(frequency_array, I[1, 1, ind, :], color=rgba)
                cbar = plt.colorbar(sm)
                cbar.ax.locator_params(nbins=4)
                cbar.set_label('$\mathrm{log}_{10}(\mu_r)$', rotation=270, labelpad=15)
                plt.xlabel('$\omega$, rad/s')
                plt.ylabel('$(\mathcal{I})_{1,1}$')
                plt.title('$\mathcal{I}$')
                if use_ROM is True:
                    plt.title('$\mathcal{I}$ ROM')

                plt.figure()
                for ind, nmur in enumerate(normalised_permeability_array):
                    rgba = cmap(nmur)
                    plt.semilogx(frequency_array, I[2, 2, ind, :], color=rgba)
                cbar = plt.colorbar(sm)
                cbar.ax.locator_params(nbins=4)
                cbar.set_label('$\mathrm{log}_{10}(\mu_r)$', rotation=270, labelpad=15)
                plt.xlabel('$\omega$, rad/s')
                plt.ylabel('$(\mathcal{I})_{2,2}$')
                plt.title('$\mathcal{I}$')
                if use_ROM is True:
                    plt.title('$\mathcal{I}$ ROM')

                plt.figure()
                for ind, nmur in enumerate(normalised_permeability_array):
                    rgba = cmap(nmur)
                    plt.semilogx(frequency_array, I[0, 1, ind, :], color=rgba)
                cbar = plt.colorbar(sm)
                cbar.ax.locator_params(nbins=4)
                cbar.set_label('$\mathrm{log}_{10}(\mu_r)$', rotation=270, labelpad=15)
                plt.xlabel('$\omega$, rad/s')
                plt.ylabel('$(\mathcal{I})_{0,1}$')
                plt.title('$\mathcal{I}$')
                if use_ROM is True:
                    plt.title('$\mathcal{I}$ ROM')

                plt.figure()
                for ind, nmur in enumerate(normalised_permeability_array):
                    rgba = cmap(nmur)
                    plt.semilogx(frequency_array, I[0, 2, ind, :], color=rgba)
                cbar = plt.colorbar(sm)
                cbar.ax.locator_params(nbins=4)
                cbar.set_label('$\mathrm{log}_{10}(\mu_r)$', rotation=270, labelpad=15)
                plt.xlabel('$\omega$, rad/s')
                plt.ylabel('$(\mathcal{I})_{0,2}$')
                plt.title('$\mathcal{I}$')
                if use_ROM is True:
                    plt.title('$\mathcal{I}$ ROM')

                plt.figure()
                for ind, nmur in enumerate(normalised_permeability_array):
                    rgba = cmap(nmur)
                    plt.semilogx(frequency_array, I[1, 2, ind, :], color=rgba)
                cbar = plt.colorbar(sm)
                cbar.ax.locator_params(nbins=4)
                cbar.set_label('$\mathrm{log}_{10}(\mu_r)$', rotation=270, labelpad=15)
                plt.xlabel('$\omega$, rad/s')
                plt.ylabel('$(\mathcal{I})_{1,2}$')
                plt.title('$\mathcal{I}$')
                if use_ROM is True:
                    plt.title('$\mathcal{I}$ ROM')

        else:

            # 3d matplotlib figures do not work well with a log scale. Thus we introduce this tick formatter.
            import matplotlib.ticker as mticker
            def log_tick_formatter(val, pos=None):
                if val == 0:
                    return '$10^0$'
                else:
                    return '$10^{' + f'{val:.0f}' + '}$'

            if use_ROM == False:
                I = self.I
                frequency_array = self.frequency_array
                permeability_array = self.permeability_array
            else:
                I = self.I_ROM
                frequency_array = self.frequency_array_ROM
                permeability_array = self.permeability_array_ROM

            xx, yy = np.meshgrid(np.log10(frequency_array), np.log10(permeability_array))
            zz = np.zeros(xx.shape)

            for ind in range(len(permeability_array)):
                zz[ind,:] = I[0, 0, ind, :]
            # fig = plt.figure()
            ax = plt.axes(projection='3d')
            if use_ROM is False:
                ax.plot_wireframe(xx, yy, zz)
            else:
                ax.plot_wireframe(xx, yy, zz, label='ML prediction')
            ax.set_xlabel('$\omega$ [rad/s]')
            ax.set_ylabel('$\mu_r$')
            ax.set_zlabel('$(\mathcal{I})_{0,0}$ [m$^3$]')

            # Here, we are setting the axis tick labels to be formatted as 10^n for neater presentation.
            startx, endx = ax.get_xlim()
            starty, endy = ax.get_ylim()
            ax.xaxis.set_ticks(np.round(np.arange(startx, endx+1, 1)))
            ax.yaxis.set_ticks(np.round(np.arange(starty, endy+1, 1)))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))

            # Option to add snapshot solutions to ROM surface plot.
            overlay_snapshots = True
            if (use_ROM is True) and (overlay_snapshots is True):
                xx_snapshots, yy_snapshots = np.meshgrid(np.log10(self.frequency_array), np.log10(self.permeability_array))
                zz = np.zeros(xx_snapshots.shape)

                for ind in range(len(self.permeability_array)):
                    zz[ind, :] = self.I[0, 0, ind, :]
                ax.scatter(np.ravel(xx_snapshots),np.ravel(yy_snapshots),np.ravel(zz),
                           color='r',
                           label='Snapshot Solutions')

            # Option to plot intersecting surface at the eddy current limit where wavelength <= object size.
            # add_eddy_current_limit = True
            if add_eddy_current_limit is True:
                max_omega = self.calc_eddy_current_limit()
                z_max = ax.get_zlim()[1]
                z_min = ax.get_zlim()[0]

                yy,zz = np.meshgrid(np.log10(self.permeability_array), np.linspace(z_min, z_max, 50))
                xx = np.ones(yy.shape)

                for ind, omega in enumerate(max_omega):
                    xx[:,ind] = np.log10(omega)

                surf = ax.plot_surface(xx,yy,zz, color='m', alpha=0.4, linewidth=0, label='Eddy current limit')
                # 3d plotting in matplotlib is still buggy. Introduces edgecolors2d and facecolors2d in order to
                # add legend entry.
                surf._facecolors2d = surf._facecolor3d
                surf._edgecolors2d = surf._edgecolor3d

            if plot_errorbars is True:
                # component 1:
                zz = np.zeros(xx.shape)
                for ind in range(len(permeability_array)):
                    zz[ind, :] = I[0, 0, ind, :]
                    for jnd in range(len(frequency_array)):
                        zz[ind, jnd] += self.error_tensors[ind, jnd, 0]
                        if zz[ind,jnd] > 1e-6:
                            zz[ind,jnd] = np.nan

                ax.plot_wireframe(xx,yy,zz, alpha=0.4,color='g', label='Upper/Lower limit')

                zz = np.zeros(xx.shape)
                for ind in range(len(permeability_array)):
                    zz[ind, :] = I[0, 0, ind, :]
                    for jnd in range(len(frequency_array)):
                        zz[ind, jnd] -= self.error_tensors[ind, jnd, 0]
                        if zz[ind,jnd] < -1e-6:
                            zz[ind,jnd] = np.nan
                ax.plot_wireframe(xx,yy,zz, alpha=0.4,color='g')

            plt.legend()
            plt.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))

    def plot_snapshot_parameters(self, eddy_current_limit=True):
        """
        James Elgy - 2022
        Function to plot out the grid of snapshot parameters used for the basis of the ROM.
        Optionally, plot additional eddy current limit where wavelength <= object size.
        :param eddy_current_limit:
        :return:
        """

        plt.figure()
        x,y = np.meshgrid(np.log10(self.permeability_array)/np.log10(self.log_base_permeability), np.log10(self.frequency_array))
        plt.scatter(x,y, label='Snapshot positions')
        plt.xlabel(f'$\log_{self.log_base_permeability}(\mu_r)$')
        plt.ylabel(f'$\log_{self.log_base_frequency}(\omega)$')

        if eddy_current_limit is True:
            max_freq_array = self.calc_eddy_current_limit()
            plt.plot(np.log10(self.permeability_array)/np.log10(self.log_base_permeability), np.log10(max_freq_array),
                     'k--',
                     label='Eddy current limit')
            plt.legend()

    def calc_eigenvalues(self, use_ROM=False):

        if use_ROM is True:
            permeability_array = self.permeability_array_ROM
            frequency_array = self.frequency_array_ROM
            N0 = self.N0_ROM
            R = self.R_ROM
            I = self.I_ROM
        else:
            permeability_array = self.permeability_array
            frequency_array = self.frequency_array
            N0 = self.N0_snapshots
            R = self.R
            I = self.I

        eig = np.zeros((3, len(permeability_array), len(frequency_array)), dtype=complex)
        for ind in range(len(permeability_array)):
            for jnd in range(len(frequency_array)):
                eig[:, ind, jnd] = np.linalg.eigvals(R[:, :, ind, jnd] + N0[ind, :, :]) + 1j * np.linalg.eigvals(I[:, :, ind, jnd])

        if use_ROM is True:
            self.eigenvalues_ROM = eig
        else:
            self.eigenvalues = eig
        return eig

    def save_results(self, prefix='', suffix=''):
        """
        James Elgy 2022:
        Function to auto generate folder structure for MPS data.

        :return:
        """

        foldername= 'Results_2d/'+ prefix
        foldername += f'{[mat for mat in self.mesh.GetMaterials() if mat!="air"]}'
        foldername += f'_nelements={self.mesh.ne}'
        foldername += f'_order={self.order}'
        foldername += f'_alpha={self.alpha}'
        foldername += f'_mur={self.permeability_array[0]:.2f}-{self.permeability_array[-1]:.2f}'
        foldername += f'_omega={self.frequency_array[0]:.2e}-{self.frequency_array[-1]:.2e}'
        foldername += f'_sigma={[self.sigma_dict[key] for key in self.sigma_dict.keys() if key != "air"]}'
        foldername += f'nsnapshots_{len(self.permeability_array)}x{len(self.frequency_array)}'
        foldername += '/'
        foldername += f'POD_mur={self.permeability_array_ROM[0]:.2f}-{self.permeability_array_ROM[-1]:.2f}'
        foldername += f'_POD_omega={self.frequency_array_ROM[0]:.2e}-{self.frequency_array_ROM[-1]:.2e}'
        foldername += f'_nROM_{len(self.permeability_array_ROM)}x{len(self.frequency_array_ROM)}'
        foldername += f'_tol_{self.theta0_truncation_tol}'
        foldername += suffix
        print('Saving Results to "' + foldername + '"')

        try:
            os.makedirs(foldername)
        except:
            print('Folder already exists.')
        try:
            np.save(foldername+'/R_snapshots.npy', self.R)
        except:
            pass
        try:
            np.save(foldername+'/I_snapshots.npy', self.I)
        except:
            pass
        try:
            np.save(foldername+'/N0_snapshots.npy', self.N0_snapshots)
        except:
            pass
        try:
            np.save(foldername+'/R_ROM.npy', self.R_ROM)
        except:
            pass
        try:
            np.save(foldername+'/I_ROM.npy', self.I_ROM)
        except:
            pass
        try:
            np.save(foldername+'/N0_ROM.npy', self.N0_ROM)
        except:
            pass
        try:
            np.save(foldername+'/error_tensors', self.error_tensors)
        except:
            pass
        try:
            np.save(foldername+'/eigenvalues', self.eigenvalues)
        except:
            pass
        try:
            np.save(foldername+'/eigenvalues_ROM', self.eigenvalues_ROM)
        except:
            pass
        try:
            np.save(foldername+'/theta1_U_truncated', self.theta1_U_truncated_array)
        except:
            pass
        try:
            np.save(foldername+'/theta1_Vh_truncated', self.theta1_Vh_truncated_array)
        except:
            pass
        try:
            np.save(foldername+'/theta1_Sigma_truncated', self.theta1_Sigma_truncated_array)
        except:
            pass
        try:
            np.save(foldername + '/theta0_U_truncated', self.theta0_U_truncated_array)
        except:
            pass
        try:
            np.save(foldername + '/theta0_Vh_truncated', self.theta0_Vh_truncated_array)
        except:
            pass
        try:
            np.save(foldername + '/theta0_Sigma_truncated', self.theta0_Sigma_truncated_array)
        except:
            pass


        with open(__file__, 'r') as f:
            with open(foldername+'/MultiPermeability.py', 'w') as fout:
                for line in f.readlines():
                    print(line, file=fout, end='')

        self.mesh.ngmesh.Save(foldername+'/ObjectVolFile.vol')

        save_all_figures(foldername, format='pdf')
        return foldername


    def _truncated_SVD(self, matrix, tol, n_modes='default'):
        """
        James Elgy - 2021
        Perform truncated singular value decomposition of matrix approx U_trunc @ S_trunc @ Vh_trunc.
        :param matrix: NxP matrix
        :param tol: float tolerance for the truncation
        :param n_modes: int override for the tolerance parameter to force a specific number of singular values.
        :return: U_trunc, S_trunc, Vh_trunc - Truncated left singular matrix, truncated singular values, and truncated
        hermitian of right singular matrix.
        """

        U, S, Vh = np.linalg.svd(matrix, full_matrices=False)  # Zeros are removed.
        S_norm = S / S[0]
        for ind, val in enumerate(S_norm):
            if val < tol:
                break
            cutoff_index = ind
        if n_modes != 'default':
            cutoff_index = n_modes - 1

        S_trunc = np.diag(S[:cutoff_index + 1])
        U_trunc = U[:, :cutoff_index + 1]
        Vh_trunc = Vh[:cutoff_index + 1, :]

        return U_trunc, S_trunc, Vh_trunc

    def _define_postprojection(self):
        """
        James Elgy - 2022
        Function to obtain post processing projection used to set the gauge for the theta0 solutions.
        Follows thesis by Zaglmayr pg 160ish.

        :return: proj - projection matrix.
        """

        # theta0_fes = HCurl(self.mesh, order=self.order, dirichlet='outer', flags={'nograds': True})
        theta0_fes = self.theta0_fes
        u, v = theta0_fes.TnT()
        m = BilinearForm(theta0_fes)
        m += u*v*dx
        m.Assemble()

        # build gradient matrix as sparse matrix (and corresponding scalar FESpace)
        gradmat, fesh1 = theta0_fes.CreateGradient()

        gradmattrans = gradmat.CreateTranspose()  # transpose sparse matrix
        math1 = gradmattrans @ m.mat @ gradmat  # multiply matrices
        math1[0, 0] += 1  # fix the 1-dim kernel
        invh1 = math1.Inverse(inverse="sparsecholesky")

        # build the Poisson projector with operator Algebra:
        proj = IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ m.mat

        return proj

    def _plot_modes(self, trunc, pred):
        """
        James Elgy - 2022
        Function to plot comparison between truncated right singular values and neural network approximation for each
        mode.
        :param trunc: Original truncated array
        :param pred: Neural network approximation.
        :return:
        """
        n_modes = trunc.shape[0]
        n_snapshots = trunc.shape[1]
        n_query = pred.shape[1]
        n_subplots = 5
        n_figures = int(np.ceil(n_modes/n_subplots))

        count = 0
        for ind in range(n_figures):

            fig, axs = plt.subplots(n_subplots, 1, sharex=True, sharey=True, squeeze=True)
            for sub in range(n_subplots):
                axs[sub].plot(np.linspace(0,n_snapshots,n_snapshots), trunc[count,:], label='original')
                axs[sub].plot(np.linspace(0,n_snapshots,n_query), pred[count,:], label='NN')
                axs[sub].set_ylabel(f'Mode {count}')
                if sub == n_subplots - 1 or count == n_modes-1:
                    axs[sub].set_xlabel('Snapshot')
                count += 1

                if count == n_modes:
                    break

            handles, labels = axs[sub].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center')
        # plt.figure()

    def _make_prediction_2D(self,mur, omega, model):

        x_query = np.asarray([np.log10(mur) / np.log10(self.log_base_permeability)]).reshape(-1,1)
        y_query = np.asarray([np.log10(omega)]).reshape(-1,1)

        # Making prediction
        scaled_query_x = (model.x_scaler.transform(x_query))
        scaled_query_y = (model.y_scaler.transform(y_query))

        xxq, yyq = np.meshgrid(scaled_query_x, scaled_query_y)
        query_array = np.asarray([np.ravel(xxq, order='F'), np.ravel(yyq, order='F')]).transpose()
        scaled_query_z = model.regression.predict(query_array)

        # Undoing normalisation
        if model.scaler != 'standard':
            z_query = model.z_scaler.inverse_transform(scaled_query_z.reshape((len(query_array), 1)))
        else:
            z_query = model.z_scaler.inverse_transform(scaled_query_z)

        return np.squeeze(z_query)

    def _make_prediction_1D(self, mur, model):
        query_point = np.log10(mur)/np.log10(self.log_base_permeability)
        model.x_query = np.asarray([[mur]])
        scaled_query = (model.x_scaler.transform(model.x_query))
        y_pred = model.regression.predict(scaled_query)
        y_pred = model.y_scaler.inverse_transform(y_pred)  # Denormalising.

        return y_pred

    def _convert_to_sparse(self):
        self.theta0_snapshots = [sp.csr_matrix(self.theta0_snapshots[:,:,0]),
                                 sp.csr_matrix(self.theta0_snapshots[:,:,1]),
                                 sp.csr_matrix(self.theta0_snapshots[:,:,2])]

        t1 = []
        # Unpacking output.
        for dim in range(3):
            t2 = []
            for ind, mu in enumerate(self.permeability_array):
                t2 += [sp.csr_matrix(np.squeeze(self.theta1_snapshots[:,dim,ind, :]))]
            t1 += [t2]
        self.theta1_snapshots = t1

class NeuralNetwork():
    """
    James Elgy - 2022
    Class to perform neural network curve fitting.
    Currenly supports 1d and 2d curve fitting using MLPregression from scikit learn.
    """

    def __init__(self, x_array, y_array, x_query, z_array='None', y_query='None'):
        self.neurons = (8,8)
        self.activation = 'tanh'
        self.tol = 1e-10
        self.alpha = 0
        self.scaler = 'standard'

        self.x_array = x_array
        self.y_array = y_array
        self.z_array = z_array
        self.x_query = x_query
        self.y_query = y_query

        if type(z_array) is str:
            self.flag_2d = False
        else:
            self.flag_2d = True

    def neural_network_1D(self, optimise=False):
        """
        James Elgy - 2021
        Function to use a simple neural network with one input and one output neuron for regression.
        Uses Scikit_learn mlpregressor.
        The performance of the curve fitting is not evaluated and all x_array and y_array is used.
        :param x_array: array of original datapoints (Real)
        :param y_array: array of original responses (Real)
        :param x_query: array of query points. (Real)
        :return: y_query: array of predicted responses.
        :return: regressor: The regression object.
        """

        # Normalising input data to mean=0, std=1.
        x_scaler = pp.StandardScaler()
        x_scaled = x_scaler.fit_transform(self.x_array)
        y_scaler = pp.StandardScaler()
        y_scaled = y_scaler.fit_transform(self.y_array)

        # Building and training network.
        if optimise is False:
            regressor = nn.MLPRegressor(hidden_layer_sizes=self.neurons, max_iter=5000000, activation=self.activation,
                                        solver='lbfgs', tol=self.tol, alpha=self.alpha, verbose=False, warm_start=True,
                                        random_state=None, n_iter_no_change=1000, max_fun=1000000)
        else:
            regressor = self.grid_search()
        regression = regressor.fit(x_scaled, np.ravel(y_scaled))

        # Making prediction
        scaled_query = (x_scaler.transform(self.x_query))
        y_pred = regressor.predict(scaled_query)
        y_pred = y_scaler.inverse_transform(y_pred)  # Denormalising.

        self.regression = regression
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

        return y_pred

    def neural_network_2D(self, optimise=False):
        """
        James Elgy - 2022
        Function to use a simple neural network with two input and one output neuron for regression.
        Uses Scikit_learn mlpregressor.
        The performance of the curve fitting is not evaluated and all x_array and y_array is used.
        :param x_array: array of original datapoints dim1 (Real)
        :param y_array: array of original datapoints dim2 (Real)
        :param z_array: array of original responses (Real) has shape [len(x_array)*len(y_array),]
        :param x_query: array of query points dim1. (Real)
        :param y_query: array of query points dim2. (Real)
        :return: z_query: array of predicted responses.
        :return: regressor: The regression object.
        """
        if self.scaler == 'standard':
            # Normalising input data to mean=0, std=1.
            x_scaler = pp.StandardScaler()
            y_scaler = pp.StandardScaler()
            z_scaler = pp.StandardScaler()
        else:
            x_scaler = pp.MinMaxScaler()
            y_scaler = pp.MinMaxScaler()
            z_scaler = pp.MinMaxScaler()

        x_scaled = x_scaler.fit_transform(self.x_array)
        y_scaled = y_scaler.fit_transform(self.y_array)
        z_scaled = z_scaler.fit_transform(self.z_array)

        if optimise is False:
            regressor = nn.MLPRegressor(hidden_layer_sizes=self.neurons, max_iter=500000, activation=self.activation,
                                        solver='lbfgs', tol=self.tol, alpha=self.alpha, verbose=False, warm_start=False,
                                        random_state=None, n_iter_no_change=10000, max_fun=100000)
        else:
            regressor = self.grid_search()

        # constucting input array and training model
        total_samples = len(x_scaled) * len(y_scaled)
        xx, yy = np.meshgrid(x_scaled, y_scaled)
        input_array = np.asarray([np.ravel(xx, order='F'), np.ravel(yy, order='F')]).transpose()
        regression = regressor.fit(input_array, z_scaled.flatten())

        # Making prediction
        scaled_query_x = (x_scaler.transform(self.x_query))
        scaled_query_y = (y_scaler.transform(self.y_query))

        xxq, yyq = np.meshgrid(scaled_query_x, scaled_query_y)
        query_array = np.asarray([np.ravel(xxq, order='F'), np.ravel(yyq, order='F')]).transpose()
        scaled_query_z = regression.predict(query_array)

        # Undoing normalisation
        if self.scaler != 'standard':
            z_query = z_scaler.inverse_transform(scaled_query_z.reshape((len(query_array), 1)))
        else:
            z_query = z_scaler.inverse_transform(scaled_query_z)

        # plt.figure()
        # plt.plot(z_query, label='query')
        # plt.plot(np.ravel(self.z_array), label='input')
        # plt.legend()
        # plt.xlabel('Snapshot Number')
        # plt.ylabel('$V^H$ mode 1')
        self.regression = regression
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.z_scaler = z_scaler

        return np.squeeze(z_query)

    def grid_search(self):
        parameter_space = {
            'hidden_layer_sizes': [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64),
                                   (1,), (2,), (4,), (8,), (16,), (32,), (64,),
                                   (1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 32, 32),
                                   (64, 64, 64)],
            'activation': ['tanh', 'logistic'],
            'solver': ['lbfgs'],
            'alpha': [0],
        }

        regressor = nn.MLPRegressor(random_state=None, n_iter_no_change=1000, max_fun=100000, max_iter=500000)
        clf = GridSearchCV(regressor, parameter_space, n_jobs=-1, cv=100, scoring='neg_root_mean_squared_error',
                           verbose=2)
        return clf

        # Best parameter set
        print('Best parameters found:\n', clf.best_params_)

        # All results
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    def run(self, optimisation=False):

        if self.flag_2d == True:
            z_query = self.neural_network_2D(optimise=optimisation)
            return z_query
        else:
            y_query = self.neural_network_1D(optimise=optimisation)
            return y_query

class PODP_ROM():

    def __init__(self, MPS):
         self.MPS = MPS

    def calc_ROM(self):

        # Prealocating results arrays:
        TensorArray = np.zeros([len(self.MPS.frequency_array_ROM),len(self.MPS.permeability_array_ROM), 9], dtype=complex)
        EigenValues = np.zeros([len(self.MPS.frequency_array_ROM), len(self.MPS.permeability_array_ROM), 3], dtype=complex)

        mu0 = 4 * np.pi * 1e-7
        fes = self.MPS.theta0_fes
        fes2 = self.MPS.theta1_fes
        ndof2 = fes2.ndof
        ndof = fes.ndof
        u, v = fes2.TnT()
        Theta_0 = GridFunction(fes)
        Theta0Sol = self.MPS.theta0_snapshots[:,0,:] # Placeholder to create vectors of the right size.
        # Currently only supports objects with constant conductivity.

        sigma_coef = [self.MPS.sigma_dict[mat] for mat in self.MPS.mesh.GetMaterials()]
        sigma = CoefficientFunction(sigma_coef)

        for ind in range(len(self.MPS.permeability_array)):
            if ind == 0:
                Theta1Sols_array = self.MPS.theta1_snapshots[:,:,ind,:]
            else:
                Theta1Sols_array = np.concatenate((Theta1Sols_array, self.MPS.theta1_snapshots[:,:,:,ind]), axis=2)


        ### Perform truncated SVD ###

        u1Truncated, s1, vh1 = np.linalg.svd(Theta1Sols_array[:, 0, :], full_matrices=False)
        u2Truncated, s2, vh2 = np.linalg.svd(Theta1Sols_array[:, 1, :], full_matrices=False)
        u3Truncated, s3, vh3 = np.linalg.svd(Theta1Sols_array[:, 2, :], full_matrices=False)
        # Print an update on progress
        print(' SVD complete      ')

        # scale the value of the modes
        s1norm = s1 / s1[0]
        s2norm = s2 / s2[0]
        s3norm = s3 / s3[0]

        # Decide where to truncate. Note that I am choosing the same number of modes for each direction.
        cutoff = len(self.MPS.permeability_array) * len(self.MPS.frequency_array)
        for i in range(len(self.MPS.permeability_array) * len(self.MPS.frequency_array)):
            if s1norm[i] < self.MPS.theta1_truncation_tol:
                if s2norm[i] < self.MPS.theta1_truncation_tol:
                    if s3norm[i] < self.MPS.theta1_truncation_tol:
                        cutoff = i
                        break

        u1Truncated = u1Truncated[:, :cutoff]
        u2Truncated = u2Truncated[:, :cutoff]
        u3Truncated = u3Truncated[:, :cutoff]

        # Here, I am copying important matrices to class attributes, such that they remain after the function has run.
        self.theta1_cutoff = cutoff
        self.sigma1Truncated = np.diag(s1[:cutoff])
        self.sigma2Truncated = np.diag(s2[:cutoff])
        self.sigma3Truncated = np.diag(s3[:cutoff])
        self.u1Truncated = u1Truncated
        self.u2Truncated = u2Truncated
        self.u3Truncated = u3Truncated

        self.g1_theta1 = np.zeros((cutoff, len(self.MPS.permeability_array_ROM), len(self.MPS.frequency_array_ROM)), dtype=complex)
        self.g2_theta1 = np.zeros((cutoff, len(self.MPS.permeability_array_ROM), len(self.MPS.frequency_array_ROM)), dtype=complex)
        self.g3_theta1 = np.zeros((cutoff, len(self.MPS.permeability_array_ROM), len(self.MPS.frequency_array_ROM)), dtype=complex)

        self.R1 = np.zeros((self.MPS.theta1_fes.ndof, len(self.MPS.permeability_array_ROM)), dtype=complex)
        self.R2 = np.zeros((self.MPS.theta1_fes.ndof, len(self.MPS.permeability_array_ROM)), dtype=complex)
        self.R3 = np.zeros((self.MPS.theta1_fes.ndof, len(self.MPS.permeability_array_ROM)), dtype=complex)

        ### Constuct bilinear forms ###

        print(' creating reduced order model', end='\r')
        nu_no_omega = mu0 * (self.MPS.alpha ** 2)

        ain = BilinearForm(fes2)
        aout = BilinearForm(fes2)
        # Here a0 is split into inside and outside the object as ain and aout
        ain += SymbolicBFI(self.MPS.inout * InnerProduct(curl(u), curl(v)))

        aout += SymbolicBFI((1 - self.MPS.inout) * InnerProduct(curl(u), curl(v)))
        aout += SymbolicBFI((1j)*(1 - self.MPS.inout) * self.MPS.epsi * InnerProduct(u, v))
        # a1 remains the same
        a1 = BilinearForm(fes2)
        a1 += SymbolicBFI((1j) * self.MPS.inout * nu_no_omega * sigma * InnerProduct(u, v))

        ain.Assemble()
        aout.Assemble()
        a1.Assemble()

        Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, 0]
        r1 = LinearForm(fes2)
        r1 += SymbolicLFI(self.MPS.inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0, v))
        r1 += SymbolicLFI(self.MPS.inout * (-1j) * nu_no_omega * sigma * InnerProduct(self.MPS.xivec[0], v))
        r1.Assemble()
        read_vec = r1.vec.CreateVector()
        write_vec = r1.vec.CreateVector()

        # I calculate r1, r2, and r3 for each value of permeability. Therefore I assign A1H, A0outH, and A0H 3 times.
        A0H_1 = np.zeros([ndof2, cutoff], dtype=complex)
        A0outH_1 = np.zeros([ndof2, cutoff], dtype=complex)
        A0H_2 = np.zeros([ndof2, cutoff], dtype=complex)
        A0outH_2 = np.zeros([ndof2, cutoff], dtype=complex)
        A0H_3 = np.zeros([ndof2, cutoff], dtype=complex)
        A0outH_3 = np.zeros([ndof2, cutoff], dtype=complex)
        A1H_1 = np.zeros([ndof2, cutoff], dtype=complex)
        A1H_2 = np.zeros([ndof2, cutoff], dtype=complex)
        A1H_3 = np.zeros([ndof2, cutoff], dtype=complex)

        # Populating matrices:
        for i in range(cutoff):
            read_vec.FV().NumPy()[:] = u1Truncated[:, i]
            write_vec.data = ain.mat * read_vec
            A0H_1[:, i] = write_vec.FV().NumPy()
            write_vec.data = aout.mat * read_vec
            A0outH_1[:, i] = write_vec.FV().NumPy()
            write_vec.data = a1.mat * read_vec
            A1H_1[:, i] = write_vec.FV().NumPy()
        HA0H1 = (np.conjugate(np.transpose(u1Truncated)) @ A0H_1)
        HA0outH1 = (np.conjugate(np.transpose(u1Truncated)) @ A0outH_1)
        HA1H1 = (np.conjugate(np.transpose(u1Truncated)) @ A1H_1)

        for i in range(cutoff):
            read_vec.FV().NumPy()[:] = u2Truncated[:, i]
            write_vec.data = ain.mat * read_vec
            A0H_2[:, i] = write_vec.FV().NumPy()
            write_vec.data = aout.mat * read_vec
            A0outH_2[:, i] = write_vec.FV().NumPy()
            write_vec.data = a1.mat * read_vec
            A1H_2[:, i] = write_vec.FV().NumPy()
        HA0H2 = (np.conjugate(np.transpose(u2Truncated)) @ A0H_2)
        HA0outH2 = (np.conjugate(np.transpose(u2Truncated)) @ A0outH_2)
        HA1H2 = (np.conjugate(np.transpose(u2Truncated)) @ A1H_2)

        for i in range(cutoff):
            read_vec.FV().NumPy()[:] = u3Truncated[:, i]
            write_vec.data = ain.mat * read_vec
            A0H_3[:, i] = write_vec.FV().NumPy()
            write_vec.data = aout.mat * read_vec
            A0outH_3[:, i] = write_vec.FV().NumPy()
            write_vec.data = a1.mat * read_vec
            A1H_3[:, i] = write_vec.FV().NumPy()
        HA0H3 = (np.conjugate(np.transpose(u3Truncated)) @ A0H_3)
        HA0outH3 = (np.conjugate(np.transpose(u3Truncated)) @ A0outH_3)
        HA1H3 = (np.conjugate(np.transpose(u3Truncated)) @ A1H_3)

        # Putting matrices into class atributes for easy access.
        # self.A0H1 = A0H_1
        # self.A0H2 = A0H_2
        # self.A0H3 = A0H_3
        # self.A0H3 = A0H_3
        # self.A0H3 = A0H_3
        # self.A0H3 = A0H_3
        # self.A1H1 = A1H_1
        # self.A1H2 = A1H_2
        # self.A1H3 = A1H_3
        # self.HA1H1 = HA1H1
        # self.HA1H3 = HA1H2
        # self.HA1H2 = HA1H3
        # self.HA0H1 = HA1H1
        # self.HA0H2 = HA1H2
        # self.HA0H3 = HA1H3
        # self.HA0outH1 = HA0outH1
        # self.HA0outH2 = HA0outH2
        # self.HA0outH3 = HA0outH3


        # Produce the sweep using the lower dimensional space
        # Setup variables for calculating tensors
        Theta_0j = GridFunction(fes)
        Theta_1i = GridFunction(fes2)
        Theta_1j = GridFunction(fes2)

        N0_array = np.zeros((3, 3, len(self.MPS.permeability_array_ROM)))
        R_array = np.zeros((3, 3, len(self.MPS.frequency_array_ROM), len(self.MPS.permeability_array_ROM)))
        I_array = np.zeros((3, 3, len(self.MPS.frequency_array_ROM), len(self.MPS.permeability_array_ROM)))
        count = 0

        ### EVALUATE THE ROM ###
        N0 = np.zeros([3, 3])

        theta0 = self.calc_theta0_ROM()
        theta1_solutions = np.zeros((self.MPS.theta1_fes.ndof, 3, len(self.MPS.permeability_array_ROM), len(self.MPS.frequency_array_ROM)), dtype=complex)
        count = 0
        for ind, mur in enumerate(self.MPS.permeability_array_ROM):
            # In the absence of a theta0 ROM, we simply calculate theta0 directly for each value of mur.
            mur_dict = {'air': 1.0, self.MPS.object_name: mur}
            mu = CoefficientFunction([mur_dict[mat] for mat in self.MPS.mesh.GetMaterials()])

            # Setup the grid functions and array which will be used to save
            Theta0i = GridFunction(fes)
            Theta0j = GridFunction(fes)
            Theta0Sol = np.zeros([ndof, 3])

            # Run in three directions and save in an array for later
            # for i in range(3):
            #     Theta0Sol[:, i] = Theta0(fes,
            #                              self.MPS.order,
            #                              self.MPS.alpha,
            #                              mu,
            #                              self.MPS.inout,
            #                              self.MPS.evec[i],
            #                              self.MPS.Tolerance,
            #                              self.MPS.Maxsteps,
            #                              self.MPS.epsi,
            #                              i + 1,
            #                              self.MPS.Solver)
            #
            # print(' solved theta0 problems   ')
            Theta0Sol = theta0[:,ind,:]
            # Calculate the N0 tensor
            calculate_tensor_coeffs = False
            if calculate_tensor_coeffs is True:
                VolConstant = Integrate(1 - mu ** (-1), self.MPS.mesh)
                for i in range(3):
                    Theta0i.vec.FV().NumPy()[:] = Theta0Sol[:, i]
                    for j in range(i + 1):
                        Theta0j.vec.FV().NumPy()[:] = Theta0Sol[:, j]
                        if i == j:
                            N0[i, j] = (self.MPS.alpha ** 3) * (VolConstant + (1 / 4) * (
                                Integrate(mu ** (-1) * (InnerProduct(curl(Theta0i), curl(Theta0j))), self.MPS.mesh)))
                        else:
                            N0[i, j] = (self.MPS.alpha ** 3 / 4) * (
                                Integrate(mu ** (-1) * (InnerProduct(curl(Theta0i), curl(Theta0j))), self.MPS.mesh))

                # Copy the tensor
                N0 += np.transpose(N0 - np.eye(3) @ N0)
                N0_array[:, :, ind] = N0

            # Returning to the rom component of the code.

            # For testing purposes, I have chosen to recalculate r for each value of permeability. This will probably
            # replace the same section of code earlier in the script. For now, I have simply copied the same code.
            Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, 0]
            r1 = LinearForm(fes2)
            r1 += SymbolicLFI(self.MPS.inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0, v))
            r1 += SymbolicLFI(self.MPS.inout * (-1j) * nu_no_omega * sigma * InnerProduct(self.MPS.xivec[0], v))
            r1.Assemble()
            read_vec = r1.vec.CreateVector()
            write_vec = r1.vec.CreateVector()

            Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, 1]
            r2 = LinearForm(fes2)
            r2 += SymbolicLFI(self.MPS.inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0, v))
            r2 += SymbolicLFI(self.MPS.inout * (-1j) * nu_no_omega * sigma * InnerProduct(self.MPS.xivec[1], v))
            r2.Assemble()

            Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, 2]
            r3 = LinearForm(fes2)
            r3 += SymbolicLFI(self.MPS.inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0, v))
            r3 += SymbolicLFI(self.MPS.inout * (-1j) * nu_no_omega * sigma * InnerProduct(self.MPS.xivec[2], v))
            r3.Assemble()

            R1 = r1.vec.FV().NumPy()[:]
            R2 = r2.vec.FV().NumPy()[:]
            R3 = r3.vec.FV().NumPy()[:]
            HR1 = (np.conjugate(np.transpose(u1Truncated)) @ np.transpose(R1))
            HR2 = (np.conjugate(np.transpose(u2Truncated)) @ np.transpose(R2))
            HR3 = (np.conjugate(np.transpose(u3Truncated)) @ np.transpose(R3))

            self.R1[:,ind] = R1
            self.R2[:,ind] = R2
            self.R3[:,ind] = R3

            for k, omega in enumerate(self.MPS.frequency_array_ROM):
                count += 1

                # This part is for obtaining the solutions in the lower dimensional space
                print(' solving reduced order system %d/%d    ' % (count, len(self.MPS.frequency_array_ROM)*len(self.MPS.permeability_array_ROM)), end='\r')
                t1 = time.time()
                # Here, we reintroduce the dependency on mur that was removed when we first consturcted A0.
                g1 = np.linalg.solve((HA0H1 / mur) + HA0outH1 + (HA1H1 * omega), HR1 * omega)
                g2 = np.linalg.solve((HA0H2 / mur) + HA0outH2 + (HA1H2 * omega), HR2 * omega)
                g3 = np.linalg.solve((HA0H3 / mur) + HA0outH3 + (HA1H3 * omega), HR3 * omega)

                # I am saving g solutions as class attributes for access with error calculation.
                self.g1_theta1[:,ind,k] = g1
                self.g2_theta1[:,ind,k] = g2
                self.g3_theta1[:,ind,k] = g3

                # This part projects the problem to the higher dimensional space
                W1 = np.dot(u1Truncated, g1).flatten()
                W2 = np.dot(u2Truncated, g2).flatten()
                W3 = np.dot(u3Truncated, g3).flatten()

                theta1_solutions[:, 0, ind, k] = W1
                theta1_solutions[:, 1, ind, k] = W2
                theta1_solutions[:, 2, ind, k] = W3

                if calculate_tensor_coeffs is True:
                    # Calculate the tensors
                    nu = omega * mu0 * (self.MPS.alpha ** 2)
                    R = np.zeros([3, 3])
                    I = np.zeros([3, 3])

                    for i in range(3):
                        Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, i]
                        xii = self.MPS.xivec[i]
                        if i == 0:
                            Theta_1i.vec.FV().NumPy()[:] = W1
                        if i == 1:
                            Theta_1i.vec.FV().NumPy()[:] = W2
                        if i == 2:
                            Theta_1i.vec.FV().NumPy()[:] = W3
                        for j in range(i + 1):
                            Theta_0j.vec.FV().NumPy()[:] = Theta0Sol[:, j]
                            xij = self.MPS.xivec[j]
                            if j == 0:
                                Theta_1j.vec.FV().NumPy()[:] = W1
                            if j == 1:
                                Theta_1j.vec.FV().NumPy()[:] = W2
                            if j == 2:
                                Theta_1j.vec.FV().NumPy()[:] = W3

                            # Real and Imaginary parts
                            R[i, j] = -(((self.MPS.alpha ** 3) / 4) * Integrate(
                                (mu ** (-1)) * (curl(Theta_1j) * Conj(curl(Theta_1i))),
                                self.MPS.mesh)).real
                            I[i, j] = ((self.MPS.alpha ** 3) / 4) * Integrate(
                                self.MPS.inout * nu * self.MPS.sigma_dict['sphere'] * ((Theta_1j + Theta_0j + xij) * (Conj(Theta_1i) + Theta_0 + xii)),
                                self.MPS.mesh).real

                    R += np.transpose(R - np.diag(np.diag(R))).real
                    I += np.transpose(I - np.diag(np.diag(I))).real

                    R_array[:, :, k, ind] = R
                    I_array[:, :, k, ind] = I

                    # Save in arrays
                    TensorArray[k, ind, :] = (N0 + R + 1j * I).flatten()
                    EigenValues[k, ind, :] = np.sort(np.linalg.eigvals(N0 + R)) + 1j * np.sort(np.linalg.eigvals(I))
        if calculate_tensor_coeffs is True:
            self.TensorArray = TensorArray
            self.EigenValues = EigenValues
        else:

            self.MPS.theta1_ROM = theta1_solutions
            return W1, W2, W3

    def calc_theta0_ROM(self):
        Theta0Sols_array = self.MPS.theta0_snapshots

        self.g1_theta0 = np.asarray([])
        self.g2_theta0 = np.asarray([])
        self.g3_theta0 = np.asarray([])
        mu0 = 4 * np.pi * 1e-7
        fes = self.MPS.theta0_fes
        ndof = fes.ndof
        u, v = fes.TnT()
        Theta_0 = GridFunction(fes)

        u1Truncated, s1, vh1 = np.linalg.svd(Theta0Sols_array[:, :, 0], full_matrices=False)
        u2Truncated, s2, vh2 = np.linalg.svd(Theta0Sols_array[:, :, 1], full_matrices=False)
        u3Truncated, s3, vh3 = np.linalg.svd(Theta0Sols_array[:, :, 2], full_matrices=False)
        # Print an update on progress
        print(' SVD complete      ')

        # scale the value of the modes
        s1norm = s1 / s1[0]
        s2norm = s2 / s2[0]
        s3norm = s3 / s3[0]

        # Decide where to truncate. Note that I am choosing the same number of modes for each direction.
        cutoff = len(self.MPS.permeability_array)
        for i in range(len(self.MPS.permeability_array)):
            if s1norm[i] < self.MPS.theta0_truncation_tol:
                if s2norm[i] < self.MPS.theta0_truncation_tol:
                    if s3norm[i] < self.MPS.theta0_truncation_tol:
                        cutoff = i
                        break

        u1Truncated = u1Truncated[:, :cutoff]
        u2Truncated = u2Truncated[:, :cutoff]
        u3Truncated = u3Truncated[:, :cutoff]

        self.u1Truncated_theta0 = u1Truncated
        self.u2Truncated_theta0 = u2Truncated
        self.u3Truncated_theta0 = u3Truncated
        self.theta0_cutoff = cutoff
        self.sigma1Truncated_theta0 = np.diag(s1[:cutoff])
        self.sigma2Truncated_theta0 = np.diag(s2[:cutoff])
        self.sigma3Truncated_theta0 = np.diag(s3[:cutoff])

        print(' creating reduced order model', end='\r')
        # nu_no_omega = mu0 * (self.MPS.alpha ** 2)

        ain = BilinearForm(fes)
        aout = BilinearForm(fes)
        # Here a0 is split into inside and outside the object as ain and aout
        # From eqn 9 in efficient comp paper. This also corresponds to the A0 component of eqn 24.
        ain += SymbolicBFI(self.MPS.inout * InnerProduct(curl(u), Conj(curl(v))))
        ain += SymbolicBFI(self.MPS.inout * self.MPS.epsi * InnerProduct(u,Conj(v)))
        aout += SymbolicBFI((1 - self.MPS.inout) * InnerProduct(curl(u), Conj(curl(v))))
        aout += SymbolicBFI((1 - self.MPS.inout) * self.MPS.epsi * InnerProduct(u, Conj(v)))

        ain.Assemble()
        aout.Assemble()

        # Calculating rhs for first dimension. The rhs is split into an inside and outside component. This is so that
        # we can multiply r1in by (2*(1-1/mu)) later when we solve the ROM.
        # Theta_0.vec.FV().NumPy()[:] = Theta0Sols_array[:, :, 0]
        r1in = LinearForm(fes)
        r1in += SymbolicLFI(self.MPS.inout * InnerProduct(self.MPS.evec[0], Conj(curl(v))))
        r1in.Assemble()

        r2in = LinearForm(fes)
        r2in += SymbolicLFI(self.MPS.inout * InnerProduct(self.MPS.evec[1], Conj(curl(v))))
        r2in.Assemble()

        r3in = LinearForm(fes)
        r3in += SymbolicLFI(self.MPS.inout * InnerProduct(self.MPS.evec[2], Conj(curl(v))))
        r3in.Assemble()

        read_vec = r1in.vec.CreateVector() # Preallocating size
        write_vec = r1in.vec.CreateVector()

        # I calculate r1, r2, and r3 for each value of permeability. Therefore I assign A1H, A0outH, and A0H 3 times.
        A0H_1 = np.zeros([ndof, cutoff])
        A0outH_1 = np.zeros([ndof, cutoff])
        A0H_2 = np.zeros([ndof, cutoff])
        A0outH_2 = np.zeros([ndof, cutoff])
        A0H_3 = np.zeros([ndof, cutoff])
        A0outH_3 = np.zeros([ndof, cutoff])

        # Populating matrices:
        for i in range(cutoff):
            read_vec.FV().NumPy()[:] = u1Truncated[:, i]
            write_vec.data = ain.mat * read_vec
            A0H_1[:, i] = write_vec.FV().NumPy()
            write_vec.data = aout.mat * read_vec
            A0outH_1[:, i] = write_vec.FV().NumPy()
        HA0H1 = (np.conjugate(np.transpose(u1Truncated)) @ A0H_1)
        HA0outH1 = (np.conjugate(np.transpose(u1Truncated)) @ A0outH_1)


        for i in range(cutoff):
            read_vec.FV().NumPy()[:] = u2Truncated[:, i]
            write_vec.data = ain.mat * read_vec
            A0H_2[:, i] = write_vec.FV().NumPy()
            write_vec.data = aout.mat * read_vec
            A0outH_2[:, i] = write_vec.FV().NumPy()
        HA0H2 = (np.conjugate(np.transpose(u2Truncated)) @ A0H_2)
        HA0outH2 = (np.conjugate(np.transpose(u2Truncated)) @ A0outH_2) # This is (U^m)^H @ A @ U^m


        for i in range(cutoff):
            read_vec.FV().NumPy()[:] = u3Truncated[:, i]
            write_vec.data = ain.mat * read_vec
            A0H_3[:, i] = write_vec.FV().NumPy()
            write_vec.data = aout.mat * read_vec
            A0outH_3[:, i] = write_vec.FV().NumPy()
        HA0H3 = (np.conjugate(np.transpose(u3Truncated)) @ A0H_3)
        HA0outH3 = (np.conjugate(np.transpose(u3Truncated)) @ A0outH_3)


        # This is the bit where we solve the ROM
        count = 0
        theta0_solutions = np.zeros((ndof, len(self.MPS.permeability_array_ROM), 3))
        for ind, mur in enumerate(self.MPS.permeability_array_ROM):
            #
            mur_dict = {'air': 1.0, self.MPS.object_name: mur}
            mu = CoefficientFunction([mur_dict[mat] for mat in self.MPS.mesh.GetMaterials()])

            count += 1
            print(count)

            # Constructing rhs for theta0 problem.
            R1in = r1in.vec.FV().NumPy()
            # R1out = r1out.vec.FV().NumPy()
            R2in = r2in.vec.FV().NumPy()
            # R2out = r2out.vec.FV().NumPy()
            R3in = r3in.vec.FV().NumPy()
            # R3out = r3out.vec.FV().NumPy()

            HR1in = (np.conjugate(np.transpose(u1Truncated)) @ np.transpose(R1in))
            HR2in = (np.conjugate(np.transpose(u2Truncated)) @ np.transpose(R2in))
            HR3in = (np.conjugate(np.transpose(u3Truncated)) @ np.transpose(R3in))
            # I shouldn't need to calculate r outside, since mur(outside)=1 and the factor (2*(1-1/1))=0.
            # I've left it in at the moment for clarity.
            # HR1out = (np.conjugate(np.transpose(u1Truncated)) @ np.transpose(R1out))
            # HR2out = (np.conjugate(np.transpose(u2Truncated)) @ np.transpose(R2out))
            # HR3out = (np.conjugate(np.transpose(u3Truncated)) @ np.transpose(R3out))



            # This part is for obtaining the solutions in the lower dimensional space
            print(' solving reduced order system %d/%d    ' % (count, len(self.MPS.permeability_array_ROM)), end='\r')
            t1 = time.time()
            # Here, we reintroduce the dependency on mur that was removed when we first consturcted A0.
            g1 = np.linalg.solve((HA0H1 / mur) + HA0outH1, (2*(1-1/mur)*HR1in))
            g2 = np.linalg.solve((HA0H2 / mur) + HA0outH2, (2*(1-1/mur)*HR2in))
            g3 = np.linalg.solve((HA0H3 / mur) + HA0outH3, (2*(1-1/mur)*HR3in))

            # I am saving g solutions as class attributes for access with error calculation.
            if ind == 0:
                self.g1_theta0 = g1
                self.g2_theta0 = g2
                self.g3_theta0 = g3
            else:
                self.g1_theta0 = np.append(self.g1_theta0, g1)
                self.g2_theta0 = np.append(self.g2_theta0, g2)
                self.g3_theta0 = np.append(self.g3_theta0, g3)

            # This part projects the problem to the higher dimensional space
            W1 = np.dot(u1Truncated, g1).flatten()
            W2 = np.dot(u2Truncated, g2).flatten()
            W3 = np.dot(u3Truncated, g3).flatten()

            theta0_solutions[:,count-1,0] = W1
            theta0_solutions[:,count-1,1] = W2
            theta0_solutions[:,count-1,2] = W3

        self.MPS.theta0_ROM = theta0_solutions
        return theta0_solutions

    def eval_single_parameter_pair(self, x):

        TensorArray = np.zeros([1, 9], dtype=complex)
        EigenValues = np.zeros([1, 3], dtype=complex)

        mu0 = 4 * np.pi * 1e-7
        fes = self.MPS.theta0_fes
        fes2 = self.MPS.theta1_fes
        ndof2 = fes2.ndof
        ndof = fes.ndof
        u, v = fes2.TnT()
        Theta_0 = GridFunction(fes)
        Theta0Sol = self.MPS.theta0_snapshots[:,0,:] # Placeholder to create vectors of the right size.

        # Unpacking
        HA1H1 = self.HA1H1
        HA1H3 = self.HA1H2
        HA1H2 = self.HA1H3
        HA0H1 = self.HA1H1
        HA0H2 = self.HA1H2
        HA0H3 = self.HA1H3
        HA0outH1 = self.HA0outH1
        HA0outH2 = self.HA0outH2
        HA0outH3 = self.HA0outH3
        u1Truncated = self.u1Truncated
        u2Truncated = self.u2Truncated
        u3Truncated = self.u3Truncated


        # Produce the sweep using the lower dimensional space
        # Setup variables for calculating tensors
        Theta_0j = GridFunction(fes)
        Theta_1i = GridFunction(fes2)
        Theta_1j = GridFunction(fes2)

        N0_array = np.zeros((3, 3, 1))
        R_array = np.zeros((3, 3, 1))
        I_array = np.zeros((3, 3, 1))
        count = 0

        ### EVALUATE THE ROM ###
        N0 = np.zeros([3, 3])

        for ind, mur in enumerate(self.MPS.permeability_array_ROM):
            # In the absence of a theta0 ROM, we simply calculate theta0 directly for each value of mur.
            mur_dict = {'air': 1.0, self.MPS.object_name: mur}
            mu = CoefficientFunction([mur_dict[mat] for mat in self.MPS.mesh.GetMaterials()])

            # Setup the grid functions and array which will be used to save
            Theta0i = GridFunction(fes)
            Theta0j = GridFunction(fes)
            Theta0Sol = np.zeros([ndof, 3])

            # Run in three directions and save in an array for later
            for i in range(3):
                Theta0Sol[:, i] = Theta0(fes,
                                         self.MPS.order,
                                         self.MPS.alpha,
                                         mu,
                                         self.MPS.inout,
                                         self.MPS.evec[i],
                                         self.MPS.Tolerance,
                                         self.MPS.Maxsteps,
                                         self.MPS.epsi,
                                         i + 1,
                                         self.MPS.Solver)

            print(' solved theta0 problems   ')

            # Calculate the N0 tensor
            VolConstant = Integrate(1 - mu ** (-1), self.MPS.mesh)
            for i in range(3):
                Theta0i.vec.FV().NumPy()[:] = Theta0Sol[:, i]
                for j in range(i + 1):
                    Theta0j.vec.FV().NumPy()[:] = Theta0Sol[:, j]
                    if i == j:
                        N0[i, j] = (self.MPS.alpha ** 3) * (VolConstant + (1 / 4) * (
                            Integrate(mu ** (-1) * (InnerProduct(curl(Theta0i), curl(Theta0j))), self.MPS.mesh)))
                    else:
                        N0[i, j] = (self.MPS.alpha ** 3 / 4) * (
                            Integrate(mu ** (-1) * (InnerProduct(curl(Theta0i), curl(Theta0j))), self.MPS.mesh))

            # Copy the tensor
            N0 += np.transpose(N0 - np.eye(3) @ N0)

            N0_array[:, :, ind] = N0
            # Returning to the rom component of the code.

            # For testing purposes, I have chosen to recalculate r for each value of permeability. This will probably
            # replace the same section of code earlier in the script. For now, I have simply copied the same code.
            Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, 0]
            r1 = LinearForm(fes2)
            r1 += SymbolicLFI(
                self.MPS.inout * (-1j) * nu_no_omega * self.MPS.sigma_dict['sphere'] * InnerProduct(Theta_0, v))
            r1 += SymbolicLFI(
                self.MPS.inout * (-1j) * nu_no_omega * self.MPS.sigma_dict['sphere'] * InnerProduct(self.MPS.xivec[0],
                                                                                                    v))
            r1.Assemble()
            read_vec = r1.vec.CreateVector()
            write_vec = r1.vec.CreateVector()

            Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, 1]
            r2 = LinearForm(fes2)
            r2 += SymbolicLFI(
                self.MPS.inout * (-1j) * nu_no_omega * self.MPS.sigma_dict['sphere'] * InnerProduct(Theta_0, v))
            r2 += SymbolicLFI(
                self.MPS.inout * (-1j) * nu_no_omega * self.MPS.sigma_dict['sphere'] * InnerProduct(self.MPS.xivec[1],
                                                                                                    v))
            r2.Assemble()

            Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, 2]
            r3 = LinearForm(fes2)
            r3 += SymbolicLFI(
                self.MPS.inout * (-1j) * nu_no_omega * self.MPS.sigma_dict['sphere'] * InnerProduct(Theta_0, v))
            r3 += SymbolicLFI(
                self.MPS.inout * (-1j) * nu_no_omega * self.MPS.sigma_dict['sphere'] * InnerProduct(self.MPS.xivec[2], v))
            r3.Assemble()

            R1 = r1.vec.FV().NumPy()
            R2 = r2.vec.FV().NumPy()
            R3 = r3.vec.FV().NumPy()
            HR1 = (np.conjugate(np.transpose(u1Truncated)) @ np.transpose(R1))
            HR2 = (np.conjugate(np.transpose(u2Truncated)) @ np.transpose(R2))
            HR3 = (np.conjugate(np.transpose(u3Truncated)) @ np.transpose(R3))

            for k, omega in enumerate(self.MPS.frequency_array_ROM):
                count += 1
                print(count)

                # This part is for obtaining the solutions in the lower dimensional space
                print(' solving reduced order system %d/%d    ' % (
                count, len(self.MPS.frequency_array_ROM) * len(self.MPS.permeability_array_ROM)), end='\r')
                t1 = time.time()
                # Here, we reintroduce the dependency on mur that was removed when we first consturcted A0.
                g1 = np.linalg.solve((HA0H1 / mur) + HA0outH1 + HA1H1 * omega, HR1 * omega)
                g2 = np.linalg.solve((HA0H2 / mur) + HA0outH2 + HA1H2 * omega, HR2 * omega)
                g3 = np.linalg.solve((HA0H3 / mur) + HA0outH3 + HA1H3 * omega, HR3 * omega)

                # This part projects the problem to the higher dimensional space
                W1 = np.dot(u1Truncated, g1).flatten()
                W2 = np.dot(u2Truncated, g2).flatten()
                W3 = np.dot(u3Truncated, g3).flatten()

                # Calculate the tensors
                nu = omega * mu0 * (self.MPS.alpha ** 2)
                R = np.zeros([3, 3])
                I = np.zeros([3, 3])

                for i in range(3):
                    Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, i]
                    xii = self.MPS.xivec[i]
                    if i == 0:
                        Theta_1i.vec.FV().NumPy()[:] = W1
                    if i == 1:
                        Theta_1i.vec.FV().NumPy()[:] = W2
                    if i == 2:
                        Theta_1i.vec.FV().NumPy()[:] = W3
                    for j in range(i + 1):
                        Theta_0j.vec.FV().NumPy()[:] = Theta0Sol[:, j]
                        xij = self.MPS.xivec[j]
                        if j == 0:
                            Theta_1j.vec.FV().NumPy()[:] = W1
                        if j == 1:
                            Theta_1j.vec.FV().NumPy()[:] = W2
                        if j == 2:
                            Theta_1j.vec.FV().NumPy()[:] = W3

                        # Real and Imaginary parts
                        R[i, j] = -(((self.MPS.alpha ** 3) / 4) * Integrate(
                            (mu ** (-1)) * (curl(Theta_1j) * Conj(curl(Theta_1i))),
                            self.MPS.mesh)).real
                        I[i, j] = ((self.MPS.alpha ** 3) / 4) * Integrate(
                            self.MPS.inout * nu * self.MPS.sigma_dict['sphere'] * (
                                        (Theta_1j + Theta_0j + xij) * (Conj(Theta_1i) + Theta_0 + xii)),
                            self.MPS.mesh).real

                R += np.transpose(R - np.diag(np.diag(R))).real
                I += np.transpose(I - np.diag(np.diag(I))).real

                R_array[:, :, k, ind] = R
                I_array[:, :, k, ind] = I

                # Save in arrays
                TensorArray[k, :] = (N0 + R + 1j * I).flatten()
                EigenValues[k, :] = np.sort(np.linalg.eigvals(N0 + R)) + 1j * np.sort(np.linalg.eigvals(I))

    # def calc_error_bars(self):
    #     self.error_tensor = np.zeros((3, 3, len(self.MPS.permeability_array_ROM), len(self.MPS.frequency_array_ROM)))
    #     #Calculating stability constant alphaLB by solving simplified transmission problem for lowest frequency and
    #     # permeability of interest:
    #     Mu0 = 4*np.pi*10**(-7)
    #     mur_dict = {'air':1, 'sphere':self.MPS.permeability_array[0]}
    #     mu = CoefficientFunction([mur_dict[mat] for mat in self.MPS.mesh.GetMaterials()]) #float(self.MPS.permeability_array_ROM[0])
    #     sigma = CoefficientFunction([self.MPS.sigma_dict[mat] for mat in self.MPS.mesh.GetMaterials()])
    #     dom_nrs_metal = [0 if mat == "air" else 1 for mat in self.MPS.mesh.GetMaterials()]
    #
    #     fes0 = HCurl(self.MPS.mesh, order=0, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    #     ndof0 = fes0.ndof
    #     RerrorReduced1 = np.zeros([ndof0, self.theta1_cutoff * 3 + 1], dtype=complex)
    #     RerrorReduced2 = np.zeros([ndof0, self.theta1_cutoff * 3 + 1], dtype=complex)
    #     RerrorReduced3 = np.zeros([ndof0, self.theta1_cutoff * 3 + 1], dtype=complex)
    #     ProH = GridFunction(self.MPS.theta1_fes)
    #     ProL = GridFunction(fes0)
    #
    #
    #     fes3 = HCurl(self.MPS.mesh, order=self.MPS.order, dirichlet="outer", gradientdomains=dom_nrs_metal)
    #     ndof3 = fes3.ndof
    #     Omega = self.MPS.frequency_array[0]
    #     u, v = fes3.TnT()
    #     amax = BilinearForm(fes3)
    #     amax += (mu ** (-1)) * curl(u) * curl(v) * dx
    #     amax += (1 - self.MPS.inout) * self.MPS.epsi * u * v * dx
    #     amax += self.MPS.inout * sigma * (self.MPS.alpha ** 2) * Mu0 * Omega * u * v * dx
    #
    #     m = BilinearForm(fes3)
    #     m += u * v * dx
    #
    #     apre = BilinearForm(fes3)
    #     apre += curl(u) * curl(v) * dx + u * v * dx
    #     pre = Preconditioner(amax, "bddc")
    #
    #     with TaskManager():
    #         amax.Assemble()
    #         m.Assemble()
    #         apre.Assemble()
    #
    #         # build gradient matrix as sparse matrix (and corresponding scalar FESpace)
    #         gradmat, fesh1 = fes3.CreateGradient()
    #         gradmattrans = gradmat.CreateTranspose()  # transpose sparse matrix
    #         math1 = gradmattrans @ m.mat @ gradmat  # multiply matrices
    #         math1[0, 0] += 1  # fix the 1-dim kernel
    #         invh1 = math1.Inverse(inverse="sparsecholesky")
    #
    #         # build the Poisson projector with operator Algebra:
    #         proj = IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ m.mat
    #         projpre = proj @ pre.mat
    #         evals, evecs = solvers.PINVIT(amax.mat, m.mat, pre=projpre, num=1, maxit=50)
    #
    #     alphaLB = evals[0]
    #     print(f'Lower bound alphaLB = {alphaLB} \n')
    #
    #     # Constructing errors: preallocation:
    #     # dom_nrs_metal = [0 if mat == "air" else 1 for mat in self.MPS.mesh.GetMaterials()]
    #     fes0 = HCurl(self.MPS.mesh, order=0, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    #
    #     # Constructing A=B0 + 1/mur B1 + omega C1
    #     u, v = self.MPS.theta1_fes.TnT()
    #     B0 = BilinearForm(self.MPS.theta1_fes)
    #     B1 = BilinearForm(self.MPS.theta1_fes)
    #     C1 = BilinearForm(self.MPS.theta1_fes)
    #
    #     B0 += SymbolicBFI((1-self.MPS.inout) * InnerProduct(curl(u), curl(v)))
    #     B0 += SymbolicBFI(self.MPS.epsi * (1-self.MPS.inout) * InnerProduct(u,v))
    #     B1 += SymbolicBFI(self.MPS.inout * InnerProduct(curl(u), curl(v)))
    #     C1 += SymbolicBFI(-1j * self.MPS.alpha**2 * Mu0 * sigma * (1-self.MPS.inout) * InnerProduct(u,v))
    #
    #     B0.Assemble()
    #     B1.Assemble()
    #     C1.Assemble()
    #
    #     # Converting to scipy sparse matrices for multiplication with numpy.
    #     rows, cols, vals = B0.mat.COO()
    #     B0_sparse = sp.csr_matrix((vals,(rows,cols)))
    #     rows, cols, vals = B1.mat.COO()
    #     B1_sparse = sp.csr_matrix((vals, (rows, cols)))
    #     rows, cols, vals = C1.mat.COO()
    #     C1_sparse = sp.csr_matrix((vals, (rows, cols)))
    #
    #     del rows, cols, vals
    #
    #     # This constructs W^(i) = [r, B0 U^(m,i) Sigma^(m,i), B1 U^(m,i) Sigma^(m,i), C1 U^(m,i) Sigma^(m,i)]. Below eqn 31 in efficient comp paper.
    #
    #     RerrorReduced1_high_dimensional = self.R1[:,None] #C1_sparse @ USigma1_0
    #     RerrorReduced1_high_dimensional = np.append(RerrorReduced1_high_dimensional, B0_sparse @ self.u1Truncated, axis=1)
    #     RerrorReduced1_high_dimensional = np.append(RerrorReduced1_high_dimensional, B1_sparse @ self.u1Truncated, axis=1)
    #     RerrorReduced1_high_dimensional = np.append(RerrorReduced1_high_dimensional, C1_sparse @ self.u1Truncated, axis=1)
    #
    #     RerrorReduced2_high_dimensional = self.R2[:,None] #C1_sparse @ USigma2_0
    #     RerrorReduced2_high_dimensional = np.append(RerrorReduced2_high_dimensional, B0_sparse @ self.u2Truncated, axis=1)
    #     RerrorReduced2_high_dimensional = np.append(RerrorReduced2_high_dimensional, B1_sparse @ self.u2Truncated, axis=1)
    #     RerrorReduced2_high_dimensional = np.append(RerrorReduced2_high_dimensional, C1_sparse @ self.u2Truncated, axis=1)
    #
    #     RerrorReduced3_high_dimensional = self.R3[:,None] #C1_sparse @ USigma3_0
    #     RerrorReduced3_high_dimensional = np.append(RerrorReduced3_high_dimensional, B0_sparse @ self.u3Truncated, axis=1)
    #     RerrorReduced3_high_dimensional = np.append(RerrorReduced3_high_dimensional, B1_sparse @ self.u3Truncated, axis=1)
    #     RerrorReduced3_high_dimensional = np.append(RerrorReduced3_high_dimensional, C1_sparse @ self.u3Truncated, axis=1)
    #
    #
    #     # We now project RerrorReduced from a high order to order 0.
    #     # Set performs an element-wise L2 projection combined with arithmetic averaging of coupling dofs. This is
    #     # used to project the problem from the high order approximation to the zeroth order.
    #     temp_rerror_higher_dimensional = GridFunction(self.MPS.theta1_fes)
    #     temp_rerror_lower_dimensional = GridFunction(fes0)
    #     temp_r1 = np.zeros((fes0.ndof, RerrorReduced1_high_dimensional.shape[1]), dtype=complex)
    #     temp_r2= np.zeros((fes0.ndof, RerrorReduced1_high_dimensional.shape[1]), dtype=complex)
    #     temp_r3 = np.zeros((fes0.ndof, RerrorReduced1_high_dimensional.shape[1]), dtype=complex)
    #
    #     for ind in range(RerrorReduced1_high_dimensional.shape[1]):
    #         temp_rerror_higher_dimensional.vec.FV().NumPy()[:] = RerrorReduced1_high_dimensional[:, ind]
    #         temp_rerror_lower_dimensional.Set(temp_rerror_higher_dimensional)
    #         temp_r1[:,ind] = temp_rerror_lower_dimensional.vec.FV().NumPy()[:]
    #     RerrorReduced1_lower_dimensional = temp_r1
    #
    #     for ind in range(RerrorReduced2_high_dimensional.shape[1]):
    #         temp_rerror_higher_dimensional.vec.FV().NumPy()[:] = RerrorReduced2_high_dimensional[:, ind]
    #         temp_rerror_lower_dimensional.Set(temp_rerror_higher_dimensional)
    #         temp_r2[:,ind] = temp_rerror_lower_dimensional.vec.FV().NumPy()[:]
    #     RerrorReduced2_lower_dimensional = temp_r2
    #
    #     for ind in range(RerrorReduced3_high_dimensional.shape[1]):
    #         temp_rerror_higher_dimensional.vec.FV().NumPy()[:] = RerrorReduced3_high_dimensional[:, ind]
    #         temp_rerror_lower_dimensional.Set(temp_rerror_higher_dimensional)
    #         temp_r3[:,ind] = temp_rerror_lower_dimensional.vec.FV().NumPy()[:]
    #     RerrorReduced3_lower_dimensional = temp_r3
    #
    #     del temp_r1, temp_r2, temp_r3
    #
    #     # Constructing G_ij = W^(i) M_0^{-1} W^(j):
    #     # Constructing inverse mass matrix
    #     u, v = fes0.TnT()
    #     m = BilinearForm(fes0)
    #     m += SymbolicBFI(InnerProduct(u, v))
    #     m.Assemble()
    #     c = Preconditioner(m, "local")
    #     c.Update()
    #     inverse = CGSolver(m.mat, c.mat, precision=1e-20, maxsteps=500)
    #
    #
    #     # We are now going to construct (M_0^{-1}) W^(j).
    #     # Here, I loop over each column of RerrorReduced and do the matrix multiplication iterativly. This is
    #     # because I can't convert inverse to a sparse scipy matrix and I can't create a 2d ngsolve gridfunction.
    #     # This is the same method that Ben used for this.
    #
    #     MR1 = np.zeros((fes0.ndof, RerrorReduced1_high_dimensional.shape[1]), dtype=complex)
    #     MR2 = np.zeros((fes0.ndof, RerrorReduced1_high_dimensional.shape[1]), dtype=complex)
    #     MR3 = np.zeros((fes0.ndof, RerrorReduced1_high_dimensional.shape[1]), dtype=complex)
    #
    #     ErrorGFU = GridFunction(fes0)
    #     ProL = GridFunction(fes0)
    #     for i in range(RerrorReduced1_high_dimensional.shape[1]):
    #         # E1
    #         ProL.vec.data.FV().NumPy()[:] = RerrorReduced1_lower_dimensional[:, i]
    #         ProL.vec.data -= m.mat * ErrorGFU.vec
    #         ErrorGFU.vec.data += inverse * ProL.vec
    #         MR1[:, i] = ErrorGFU.vec.FV().NumPy()[:]
    #
    #         # E2
    #         ProL.vec.data.FV().NumPy()[:] = RerrorReduced2_lower_dimensional[:, i]
    #         ProL.vec.data -= m.mat * ErrorGFU.vec
    #         ErrorGFU.vec.data += inverse * ProL.vec
    #         MR2[:, i] = ErrorGFU.vec.FV().NumPy()[:]
    #
    #         # E3
    #         ProL.vec.data.FV().NumPy()[:] = RerrorReduced3_lower_dimensional[:, i]
    #         ProL.vec.data -= m.mat * ErrorGFU.vec
    #         ErrorGFU.vec.data += inverse * ProL.vec
    #         MR3[:, i] = ErrorGFU.vec.FV().NumPy()[:]
    #
    #     # Creating hermitian G matrix. Below eqn 31 in efficient comp paper.
    #     # inverse
    #     G11 = np.transpose(np.conjugate(RerrorReduced1_lower_dimensional)) @ MR1
    #     G22 = np.transpose(np.conjugate(RerrorReduced2_lower_dimensional)) @ MR2
    #     G33 = np.transpose(np.conjugate(RerrorReduced3_lower_dimensional)) @ MR3
    #     G12 = np.transpose(np.conjugate(RerrorReduced1_lower_dimensional)) @ MR2
    #     G13 = np.transpose(np.conjugate(RerrorReduced1_lower_dimensional)) @ MR3
    #     G23 = np.transpose(np.conjugate(RerrorReduced2_lower_dimensional)) @ MR3
    #
    #     ErrorTensors = np.zeros([len(self.MPS.permeability_array_ROM), len(self.MPS.frequency_array_ROM), 6])
    #
    #     # mu_grid = GridFunction(fes0)
    #     # mu_grid.Set(mu)
    #     for ind, mur in enumerate(self.MPS.permeability_array_ROM):
    #         mur_dict = {'air': 1, 'sphere': self.MPS.permeability_array[0]}
    #         mu = CoefficientFunction([mur_dict[mat] for mat in self.MPS.mesh.GetMaterials()])
    #         for jnd, omega in enumerate(self.MPS.frequency_array_ROM):
    #             # Constructing w^(i) from solutions to theta0 and theta1 roms.
    #
    #             w1 = np.asarray([omega])
    #             w2 = np.asarray([omega])
    #             w3 = np.asarray([omega])
    #
    #             w1 = np.append(w1, -self.g1_theta1[:,ind, jnd])
    #             w1 = np.append(w1, -(1/mur) * self.g1_theta1[:,ind,jnd])
    #             w1 = np.append(w1, -omega * self.g1_theta1[:,ind,jnd])
    #
    #             w2 = np.append(w2, -self.g2_theta1[:,ind,jnd])
    #             w2 = np.append(w2, -(1/mur) * self.g2_theta1[:,ind,jnd])
    #             w2 = np.append(w2, -omega * self.g2_theta1[:,ind,jnd])
    #
    #             w3 = np.append(w3, -self.g3_theta1[:,ind,jnd])
    #             w3 = np.append(w3, -(1/mur) * self.g3_theta1[:,ind,jnd])
    #             w3 = np.append(w3, -omega * self.g3_theta1[:,ind,jnd])
    #
    #             w1 = w1[:,None]
    #             w2 = w2[:,None]
    #             w3 = w3[:,None]
    #
    #             # We have now built W, G, and w. We therefore construct the actual error as:
    #             # ||\hat{r}_i||^2 = ( (w^(i))^H G^(i,i) W^(i) )^0.5
    #             # ||\hat{r}_i - \hat{r}_j||^2 = ( ||\hat{r}_i||^2 + ||\hat{r}_j||^2 - 2Re(( (w^(i))^H G^(i,j) W^(j) )^0.5)
    #             # THis uses the same code that Ben has already written
    #
    #             error1 = np.conjugate(np.transpose(w1)) @ G11 @ w1
    #             error2 = np.conjugate(np.transpose(w2)) @ G22 @ w2
    #             error3 = np.conjugate(np.transpose(w3)) @ G33 @ w3
    #             error12 = np.conjugate(np.transpose(w1)) @ G12 @ w2
    #             error13 = np.conjugate(np.transpose(w1)) @ G13 @ w3
    #             error23 = np.conjugate(np.transpose(w2)) @ G23 @ w3
    #
    #
    #
    #             error1 = abs(error1) ** (1 / 2)
    #             error2 = abs(error2) ** (1 / 2)
    #             error3 = abs(error3) ** (1 / 2)
    #             error12 = error12.real
    #             error13 = error13.real
    #             error23 = error23.real
    #
    #             Errors = [error1, error2, error3, error12, error13, error23]
    #
    #             for j in range(6):
    #                 if j < 3:
    #                     ErrorTensors[ind, jnd, j] = ((self.MPS.alpha ** 3) / 4) * (Errors[j] ** 2) / alphaLB
    #                 else:
    #                     ErrorTensors[ind, jnd, j] = -2 * Errors[j]
    #                     if j == 3:
    #                         ErrorTensors[ind, jnd, j] += (Errors[0] ** 2) + (Errors[1] ** 2)
    #                         ErrorTensors[ind, jnd, j] = ((self.MPS.alpha ** 3) / (8 * alphaLB)) * (
    #                                     (Errors[0] ** 2) + (Errors[1] ** 2) + ErrorTensors[ind, jnd, j])
    #                     if j == 4:
    #                         ErrorTensors[ind, jnd, j] += (Errors[0] ** 2) + (Errors[2] ** 2)
    #                         ErrorTensors[ind, jnd, j] = ((self.MPS.alpha ** 3) / (8 * alphaLB)) * (
    #                                     (Errors[0] ** 2) + (Errors[1] ** 2) + ErrorTensors[ind, jnd, j])
    #                     if j == 5:
    #                         ErrorTensors[ind, jnd, j] += (Errors[1] ** 2) + (Errors[2] ** 2)
    #                         ErrorTensors[ind, jnd, j] = ((self.MPS.alpha ** 3) / (8 * alphaLB)) * (
    #                                     (Errors[0] ** 2) + (Errors[1] ** 2) + ErrorTensors[ind, jnd, j])
    #
    #     self.error_tensor = ErrorTensors
    #
    #     return self.error_tensor

    def calc_error_bars_2(self):
        """
        James Elgy 2022
        Function to calculate upper limit error estimates for the 2D PODP ROM.
        Function follows the error derivation described in [1] with an additional mu_r dependent component.

        We define the additional splitting A = B0 + B1/mur + omega*C1 where C1 is equivalent to a1 in PODSolvers.py.
        W^(1) in [1] is then updated to a ndof x (3*M)+1 matrix and (G)_ij is updated to a 3M+1 square matrix.
        Otherwise the same as PODSolvers.py.

        [1] Wison B, Ledger P, "Efficient computation of the magnetic polarizabiltiy tensor spectral signature using
            proper orthogonal decomposition", Int. J. Numer. Methods Eng., 122(8), 1940-1963, (2021)

        :return:
        """

        # Setting material parameters and defining the 0th order fes to be used to approximate the residual.
        Mu0 = 4 * np.pi * 10 ** (-7)
        # mur_dict = {'air': 1, self.MPS.object_name: self.MPS.permeability_array[-1]}
        mur_dict = {'air': 1, self.MPS.object_name: self.MPS.permeability_array[-1]}
        mu = CoefficientFunction(
            [mur_dict[mat] for mat in self.MPS.mesh.GetMaterials()])  # float(self.MPS.permeability_array_ROM[0])
        sig = [self.MPS.sigma_dict[mat] for mat in self.MPS.mesh.GetMaterials()]
        sigma = CoefficientFunction(sig)
        dom_nrs_metal = [0 if mat == "air" else 1 for mat in self.MPS.mesh.GetMaterials()]
        fes0 = HCurl(self.MPS.mesh, order=0, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
        ndof0 = fes0.ndof
        fes2 = HCurl(self.MPS.mesh, order=self.MPS.order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
        ndof2 = fes2.ndof

        # Building B0 B1 and C1
        u, v = fes2.TnT()
        B0 = BilinearForm(self.MPS.theta1_fes)
        B1 = BilinearForm(self.MPS.theta1_fes)
        C1 = BilinearForm(self.MPS.theta1_fes)

        B0 += SymbolicBFI((1-self.MPS.inout) * InnerProduct(curl(u), curl(v)))
        B0 += SymbolicBFI(1j * self.MPS.epsi * (1-self.MPS.inout) * InnerProduct(u,v))
        B1 += SymbolicBFI(self.MPS.inout * InnerProduct(curl(u), curl(v)))
        C1 += SymbolicBFI(1j * (self.MPS.inout) * self.MPS.alpha**2 * Mu0 * sigma * InnerProduct(u,v))

        B0.Assemble()
        B1.Assemble()
        C1.Assemble()

        # Assigning R1 to ngsolve gridfuction.
        ProH = GridFunction(fes2) # ProH and ProL are the higer dimensional and lower dimensional gridfunctions.
        ProL = GridFunction(fes0)

        read_vec = GridFunction(fes2).vec
        write_vec = GridFunction(fes2).vec

        # Preallocating RerrorReduced. (W in efficient comp paper)
        RerrorReduced1 = np.zeros([ndof0,self.theta1_cutoff*3+1],dtype=complex)
        RerrorReduced2 = np.zeros([ndof0,self.theta1_cutoff*3+1],dtype=complex)
        RerrorReduced3 = np.zeros([ndof0,self.theta1_cutoff*3+1],dtype=complex)
        B0H = np.zeros([ndof2, self.theta1_cutoff], dtype=complex)
        B1H = np.zeros([ndof2, self.theta1_cutoff], dtype=complex)
        C1H = np.zeros([ndof2, self.theta1_cutoff], dtype=complex)

        ErrorTensors = np.zeros([len(self.MPS.permeability_array_ROM), len(self.MPS.frequency_array_ROM), 6])


        for ind in range(len(self.MPS.permeability_array_ROM)):
            mur = self.MPS.permeability_array_ROM[ind]
            print(f'Solving for mur = {mur}')
            # E1
            # This constructs W^(i) = [r, A0 U^(m,i), A1 U^(m,i)]. Below eqn 31 in efficient comp paper.
            ProH.vec.FV().NumPy()[:] = self.R1[:,ind]
            ProL.Set(ProH) # Set projects the high dimensional ProH to the lower dimensional vector ProL.
            RerrorReduced1[:, 0] = ProL.vec.FV().NumPy()[:]
            for i in range(self.theta1_cutoff):
                read_vec.FV().NumPy()[:] = self.u1Truncated[:, i]
                write_vec.data = B0.mat * read_vec
                B0H[:, i] = write_vec.FV().NumPy()
                write_vec.data = B1.mat * read_vec
                B1H[:, i] = write_vec.FV().NumPy()
                write_vec.data = C1.mat * read_vec
                C1H[:, i] = write_vec.FV().NumPy()

                ProH.vec.FV().NumPy()[:] = B0H[:, i]
                ProL.Set(ProH)
                RerrorReduced1[:, i + 1] = ProL.vec.FV().NumPy()[:]
                ProH.vec.FV().NumPy()[:] = B1H[:, i]
                ProL.Set(ProH)
                RerrorReduced1[:, i + self.theta1_cutoff + 1] = ProL.vec.FV().NumPy()[:]
                ProH.vec.FV().NumPy()[:] = C1H[:, i]
                ProL.Set(ProH)
                RerrorReduced1[:, i + (2*self.theta1_cutoff) + 1] = ProL.vec.FV().NumPy()[:]

            # E2
            # This constructs W^(i) = [r, A0 U^(m,i), A1 U^(m,i)]. Below eqn 31 in efficient comp paper.
            ProH.vec.FV().NumPy()[:] = self.R2[:,ind]
            ProL.Set(ProH)
            RerrorReduced2[:, 0] = ProL.vec.FV().NumPy()[:]
            for i in range(self.theta1_cutoff):
                read_vec.FV().NumPy()[:] = self.u2Truncated[:, i]
                write_vec.data = B0.mat * read_vec
                B0H[:, i] = write_vec.FV().NumPy()
                write_vec.data = B1.mat * read_vec
                B1H[:, i] = write_vec.FV().NumPy()
                write_vec.data = C1.mat * read_vec
                C1H[:, i] = write_vec.FV().NumPy()

                ProH.vec.FV().NumPy()[:] = B0H[:, i]
                ProL.Set(ProH)
                RerrorReduced2[:, i + 1] = ProL.vec.FV().NumPy()[:]
                ProH.vec.FV().NumPy()[:] = B1H[:, i]
                ProL.Set(ProH)
                RerrorReduced2[:, i + self.theta1_cutoff + 1] = ProL.vec.FV().NumPy()[:]
                ProH.vec.FV().NumPy()[:] = C1H[:, i]
                ProL.Set(ProH)
                RerrorReduced2[:, i + (2*self.theta1_cutoff) + 1] = ProL.vec.FV().NumPy()[:]

            # E3
            # This constructs W^(i) = [r, A0 U^(m,i), A1 U^(m,i)]. Below eqn 31 in efficient comp paper.
            ProH.vec.FV().NumPy()[:] = self.R3[:,ind]
            ProL.Set(ProH)
            RerrorReduced3[:, 0] = ProL.vec.FV().NumPy()[:]
            for i in range(self.theta1_cutoff):
                read_vec.FV().NumPy()[:] = self.u3Truncated[:, i]
                write_vec.data = B0.mat * read_vec
                B0H[:, i] = write_vec.FV().NumPy()
                write_vec.data = B1.mat * read_vec
                B1H[:, i] = write_vec.FV().NumPy()
                write_vec.data = C1.mat * read_vec
                C1H[:, i] = write_vec.FV().NumPy()

                ProH.vec.FV().NumPy()[:] = B0H[:, i]
                ProL.Set(ProH)
                RerrorReduced3[:, i + 1] = ProL.vec.FV().NumPy()[:]
                ProH.vec.FV().NumPy()[:] = B1H[:, i]
                ProL.Set(ProH)
                RerrorReduced3[:, i + self.theta1_cutoff + 1] = ProL.vec.FV().NumPy()[:]
                ProH.vec.FV().NumPy()[:] = C1H[:, i]
                ProL.Set(ProH)
                RerrorReduced3[:, i + (2*self.theta1_cutoff) + 1] = ProL.vec.FV().NumPy()[:]

            # We now start building MR1 MR2 and MR3 (equivalent to w)
            MR1 = np.zeros([ndof0, self.theta1_cutoff * 3 + 1], dtype=complex)
            MR2 = np.zeros([ndof0, self.theta1_cutoff * 3 + 1], dtype=complex)
            MR3 = np.zeros([ndof0, self.theta1_cutoff * 3 + 1], dtype=complex)

            # Creating inverse reduced mass matrix (0th order).
            u, v = fes0.TnT()
            m = BilinearForm(fes0)
            m += SymbolicBFI(InnerProduct(u, v))
            f = LinearForm(fes0)
            m.Assemble()
            c = Preconditioner(m, "local")
            c.Update()
            inverse = CGSolver(m.mat, c.mat, precision=1e-20, maxsteps=500)

            # Since we cannot assign a gridfunction to the entirity of RerrorReduced, we do the matrix multiplication
            # column by column to compute M_0^(-1) * W^(1).
            ErrorGFU = GridFunction(fes0)
            for i in range(3 * self.theta1_cutoff + 1):
                # E1
                ProL.vec.data.FV().NumPy()[:] = RerrorReduced1[:, i]
                ProL.vec.data -= m.mat * ErrorGFU.vec
                ErrorGFU.vec.data += inverse * ProL.vec
                MR1[:, i] = ErrorGFU.vec.FV().NumPy()

                # E2
                ProL.vec.data.FV().NumPy()[:] = RerrorReduced2[:, i]
                ProL.vec.data -= m.mat * ErrorGFU.vec
                ErrorGFU.vec.data += inverse * ProL.vec
                MR2[:, i] = ErrorGFU.vec.FV().NumPy()

                # E3
                ProL.vec.data.FV().NumPy()[:] = RerrorReduced3[:, i]
                ProL.vec.data -= m.mat * ErrorGFU.vec
                ErrorGFU.vec.data += inverse * ProL.vec
                MR3[:, i] = ErrorGFU.vec.FV().NumPy()


            # Building G_ij:
            G1 = np.transpose(np.conjugate(RerrorReduced1)) @ MR1
            G2 = np.transpose(np.conjugate(RerrorReduced2)) @ MR2
            G3 = np.transpose(np.conjugate(RerrorReduced3)) @ MR3
            G12 = np.transpose(np.conjugate(RerrorReduced1)) @ MR2
            G13 = np.transpose(np.conjugate(RerrorReduced1)) @ MR3
            G23 = np.transpose(np.conjugate(RerrorReduced2)) @ MR3

            # Clear the variables
            # RerrorReduced1, RerrorReduced2, RerrorReduced3 = None, None, None
            # MR1, MR2, MR3 = None, None, None
            # fes0, m, c, inverse = None, None, None, None

            # Calculating alphaLB
            fes3 = HCurl(self.MPS.mesh, order=self.MPS.order, dirichlet="outer", gradientdomains=dom_nrs_metal)
            ndof3 = fes3.ndof
            Omega = self.MPS.frequency_array[0]
            u, v = fes3.TnT()
            amax = BilinearForm(fes3)
            amax += (mu ** (-1)) * curl(u) * curl(v) * dx
            amax += (1 - self.MPS.inout) * self.MPS.epsi * u * v * dx
            amax += self.MPS.inout * sigma * (self.MPS.alpha ** 2) * Mu0 * Omega * u * v * dx

            m = BilinearForm(fes3)
            m += u * v * dx

            apre = BilinearForm(fes3)
            apre += curl(u) * curl(v) * dx + u * v * dx
            pre = Preconditioner(amax, "bddc")

            with TaskManager():
                amax.Assemble()
                m.Assemble()
                apre.Assemble()

                # build gradient matrix as sparse matrix (and corresponding scalar FESpace)
                gradmat, fesh1 = fes3.CreateGradient()
                gradmattrans = gradmat.CreateTranspose()  # transpose sparse matrix
                math1 = gradmattrans @ m.mat @ gradmat  # multiply matrices
                math1[0, 0] += 1  # fix the 1-dim kernel
                invh1 = math1.Inverse(inverse="sparsecholesky")

                # build the Poisson projector with operator Algebra:
                proj = IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ m.mat
                projpre = proj @ pre.mat
                evals, evecs = solvers.PINVIT(amax.mat, m.mat, pre=projpre, num=1, maxit=50)

            alphaLB = evals[0]
            print(f'Lower bound alphaLB = {alphaLB} \n')

            # Clear the variables
            # fes3, amax, apre, pre, invh1, m = None, None, None, None, None, None


            # calculating w and final error bars
            # This is the same code that Ben used, we have simply added another loop for mur.
            rom1 = np.zeros([3 * self.theta1_cutoff + 1, 1], dtype=complex)
            rom2 = np.zeros([3 * self.theta1_cutoff + 1, 1], dtype=complex)
            rom3 = np.zeros([3* self.theta1_cutoff + 1, 1], dtype=complex)


            # for ind, mur in enumerate(self.MPS.permeability_array_ROM):
            for jnd, omega in enumerate(self.MPS.frequency_array_ROM):
                g1 = self.g1_theta1[:,ind,jnd]
                g2 = self.g2_theta1[:,ind,jnd]
                g3 = self.g3_theta1[:,ind,jnd]

                rom1[0, 0] = omega
                rom2[0, 0] = omega
                rom3[0, 0] = omega

                rom1[1:1 + self.theta1_cutoff, 0] = -g1.flatten()
                rom2[1:1 + self.theta1_cutoff, 0] = -g2.flatten()
                rom3[1:1 + self.theta1_cutoff, 0] = -g3.flatten()

                rom1[1 + self.theta1_cutoff:(2*self.theta1_cutoff)+1, 0] = -(g1 / mur).flatten()
                rom2[1 + self.theta1_cutoff:(2*self.theta1_cutoff)+1, 0] = -(g2 / mur).flatten()
                rom3[1 + self.theta1_cutoff:(2*self.theta1_cutoff)+1, 0] = -(g3 / mur).flatten()

                rom1[1 + (2*self.theta1_cutoff):, 0] = -(g1 * omega).flatten()
                rom2[1 + (2*self.theta1_cutoff):, 0] = -(g2 * omega).flatten()
                rom3[1 + (2*self.theta1_cutoff):, 0] = -(g3 * omega).flatten()

                error1 = np.conjugate(np.transpose(rom1)) @ G1 @ rom1
                error2 = np.conjugate(np.transpose(rom2)) @ G2 @ rom2
                error3 = np.conjugate(np.transpose(rom3)) @ G3 @ rom3
                error12 = np.conjugate(np.transpose(rom1)) @ G12 @ rom2
                error13 = np.conjugate(np.transpose(rom1)) @ G13 @ rom3
                error23 = np.conjugate(np.transpose(rom2)) @ G23 @ rom3

                error1 = abs(error1) ** (1 / 2)
                error2 = abs(error2) ** (1 / 2)
                error3 = abs(error3) ** (1 / 2)
                error12 = error12.real
                error13 = error13.real
                error23 = error23.real

                Errors = [error1, error2, error3, error12, error13, error23]

                for j in range(6):
                    if j < 3:
                        ErrorTensors[ind, jnd, j] = ((self.MPS.alpha ** 3) / 4) * (Errors[j] ** 2) / alphaLB
                    else:
                        ErrorTensors[ind,jnd, j] = -2 * Errors[j]
                        if j == 3:
                            ErrorTensors[ind,jnd, j] += (Errors[0] ** 2) + (Errors[1] ** 2)
                            ErrorTensors[ind,jnd, j] = ((self.MPS.alpha ** 3) / (8 * alphaLB)) * (
                                        (Errors[0] ** 2) + (Errors[1] ** 2) + ErrorTensors[ind,jnd, j])
                        if j == 4:
                            ErrorTensors[ind,jnd, j] += (Errors[0] ** 2) + (Errors[2] ** 2)
                            ErrorTensors[ind,jnd, j] = ((self.MPS.alpha ** 3) / (8 * alphaLB)) * (
                                        (Errors[0] ** 2) + (Errors[1] ** 2) + ErrorTensors[ind,jnd, j])
                        if j == 5:
                            ErrorTensors[ind,jnd, j] += (Errors[1] ** 2) + (Errors[2] ** 2)
                            ErrorTensors[ind,jnd, j] = ((self.MPS.alpha ** 3) / (8 * alphaLB)) * (
                                        (Errors[0] ** 2) + (Errors[1] ** 2) + ErrorTensors[ind,jnd, j])

        self.MPS.error_tensors = ErrorTensors
        self.error_tensors = ErrorTensors
        return ErrorTensors


    def _truncated_SVD(self, matrix, tol, n_modes='default'):
        """
        James Elgy - 2021
        Perform truncated singular value decomposition of matrix approx U_trunc @ S_trunc @ Vh_trunc.
        :param matrix: NxP matrix
        :param tol: float tolerance for the truncation
        :param n_modes: int override for the tolerance parameter to force a specific number of singular values.
        :return: U_trunc, S_trunc, Vh_trunc - Truncated left singular matrix, truncated singular values, and truncated
        hermitian of right singular matrix.
        """

        U, S, Vh = np.linalg.svd(matrix, full_matrices=False)  # Zeros are removed.
        S_norm = S / S[0]
        for ind, val in enumerate(S_norm):
            if val < tol:
                break
            cutoff_index = ind
        if n_modes != 'default':
            cutoff_index = n_modes - 1

        S_trunc = np.diag(S[:cutoff_index + 1])
        U_trunc = U[:, :cutoff_index + 1]
        Vh_trunc = Vh[:cutoff_index + 1, :]

        return U_trunc, S_trunc, Vh_trunc


def eval_error_at_sample_points():
    PODP = True
    PODN = False
    RNN = False

    # Loading Mesh:
    ngmesh = ngmeshing.Mesh(dim=3)
    ngmesh.Load("../VolFiles/" + 'Claw_wodden_handle.vol')
    mesh = Mesh("../VolFiles/" + 'Claw_wodden_handle.vol')

    MPS = MultiParamSweep(0.001, 2, {'air': 0, 'ring': 17e5}, mesh)

    dom_nrs_metal = [0 if mat == "air" else 1 for mat in MPS.mesh.GetMaterials()]
    MPS.theta1_fes = HCurl(MPS.mesh, order=MPS.order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    MPS.theta0_fes = HCurl(MPS.mesh, order=MPS.order, dirichlet='outer',
                      gradientdomains=dom_nrs_metal)
    MPS.theta0_fes_complex = HCurl(MPS.mesh, order=MPS.order, dirichlet='outer', complex=True,
                           gradientdomains=dom_nrs_metal)
    # , flags = { "nograds" : True })
    # MPS.frequency_array_ROM = [100,1000,10000]
    # MPS.permeability_array_ROM = np.asarray([20, 30, 40])
    # Calculating theta0 and theta1 snapshots:
    # return MPS
    # MPS.calc_theta0_snapshots(apply_postprojection=True)
    # MPS.calc_theta1_snapshots(apply_postprojection=False)
    t0 = np.load(r"OutputResults/POD_mur=1.00-50.00_POD_omega=10.00-100000.00_nROM_32x32_tol_1e-06_PODP/theta0_snapshots.npy", allow_pickle=True)
    t1 = np.load(r"OutputResults/POD_mur=1.00-50.00_POD_omega=10.00-100000.00_nROM_32x32_tol_1e-06_PODP/theta1_snapshots.npy", allow_pickle=True)
    #
    MPS.theta0_snapshots = t0
    MPS.theta1_snapshots = t1
    MPS.calc_N0_snapshots()
    MPS.calc_R_snapshots()
    MPS.calc_I_snapshots(use_integral=False)

    # MPS.theta0_snapshots = np.zeros((MPS.theta0_fes.ndof, 16,3))
    # for dim in range(3):
    #    MPS.theta0_snapshots[:,:,dim] = t0[dim].todense()
    #
    # MPS.theta1_snapshots = np.zeros((MPS.theta0_fes.ndof, 3, 16, 16), dtype=complex)
    # for dim in range(3):
    #    for perm in range(16):
    #        MPS.theta1_snapshots[:,dim,perm,:] = t1[dim][perm].todense()
    #
    del t0, t1

    foldername = MPS.save_results()

    if RNN is True:
        RNN_start_time = time.time()
        MPS.calc_N0_snapshots()
        MPS.calc_R_snapshots()
        MPS.calc_I_snapshots()
        print('calculating Eigenvalues')
        eig_snap = np.zeros((3, len(MPS.permeability_array), len(MPS.frequency_array)), dtype=complex)
        for ind in range(len(MPS.permeability_array)):
            for jnd in range(len(MPS.frequency_array)):
                eig_snap[:, ind, jnd] = np.linalg.eigvals(MPS.R[:, :, ind, jnd] + MPS.N0_snapshots[ind, :, :]) + 1j* np.linalg.eigvals(MPS.I[:, :, ind, jnd])

        np.save(foldername + '/eigenvalues_snap.npy', eig_snap)
        # POD_RNN
        data = pd.DataFrame({'omega': [],
                             'mur': [],
                             'eig_1_real': [], 'eig_2_real': [], 'eig_3_real': [],
                             'eig_1_imag': [], 'eig_1_imag': [], 'eig_1_imag': [],
                             'N0': [], 'tensor_coeffs': []})

        for ind, mur in enumerate(MPS.permeability_array):
            EigenValues_mur = np.squeeze(eig_snap[:,ind,:]).transpose()
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
        eig_RNN = zz_real + 1j*zz_imag
        elapsed_time_rnn = time.time() - RNN_start_time
        foldername = MPS.save_results(suffix='_RNN')
        np.save(foldername + '/eigenvalues_RNN.npy', eig_RNN)
        print(f'RNN File Saved to {foldername}')

        del zz_real, zz_imag, eig_RNN, ml, output

    # PODN test:
    if PODN is True:
        print('PODN')
        start_time = time.time()
        MPS.calc_theta0_ROM()
        MPS.calc_theta1_ROM()

        MPS.calc_I_ROM()
        MPS.calc_R_ROM()
        MPS.calc_N0_ROM()
        print('calculating Eigenvalues')
        eig_podn = np.zeros((3,32,32),dtype=complex)
        for ind in range(len(MPS.permeability_array_ROM)):
           for jnd in range(len(MPS.frequency_array_ROM)):
               eig_podn[:, ind, jnd] = np.linalg.eigvals(MPS.R_ROM[:, :, ind, jnd] + MPS.N0_ROM[ind, :, :]) + 1j* np.linalg.eigvals(MPS.I_ROM[:, :, ind, jnd])
        stop_time = time.time()
        elapsed_time_podn = stop_time - start_time
        # To provide a fair timing comparison, we time from the theta snapshot solutions to the final eigenvalues for all
        # tensor coefficients. This is because the RNN method outputs takes eigenvalues and returns eigenvalues.
        foldername = MPS.save_results(suffix='_PODN')
        np.save(foldername+'/eigenvalues_PODN.npy', eig_podn)
        print(f'PODN files saved to {foldername}')

        del eig_podn, MPS.theta0_ROM, MPS.theta1_ROM, MPS.I_ROM, MPS.R_ROM

    # PODP test:
    if PODP is True:
        print('PODP')
        start_time = time.time()
        # MPS.permeability_array_ROM = np.asarray([MPS.permeability_array[-1]])
        # MPS.frequency_array_ROM = np.asarray([MPS.frequency_array[-1]])
        PODP = MPS.PODP_ROM()

        MPS.calc_I_ROM()
        MPS.calc_R_ROM()
        MPS.calc_N0_ROM()
    #     MPS.I_ROM = np.load(r"OutputResults/POD_mur=1.00-50.00_POD_omega=10.00-100000.00_nROM_32x32_tol_1e-06_PODP/I_ROM.npy", allow_pickle=True)
    #     MPS.R_ROM = np.load(r"OutputResults/POD_mur=1.00-50.00_POD_omega=10.00-100000.00_nROM_32x32_tol_1e-06_PODP/R_ROM.npy", allow_pickle=True)
    #     MPS.N0_ROM = np.load(r"OutputResults/POD_mur=1.00-50.00_POD_omega=10.00-100000.00_nROM_32x32_tol_1e-06_PODP/N0_ROM.npy", allow_pickle=True)
    # #
        print('calculating Eigenvalues')
        eig_podp = np.zeros((3, 32, 32), dtype=complex)
        for ind in range(len(MPS.permeability_array_ROM)):
            for jnd in range(len(MPS.frequency_array_ROM)):
                eig_podp[:, ind, jnd] = np.linalg.eigvals(MPS.R_ROM[:, :, ind, jnd] + MPS.N0_ROM[ind, :, :]) + 1j* np.linalg.eigvals(MPS.I_ROM[:, :, ind, jnd])
        stop_time = time.time()
        elapsed_time_podp = stop_time - start_time

        foldername = MPS.save_results(suffix='_PODP')
        np.save(foldername + '/eigenvalues_PODP.npy', eig_podp)
        print(f'PODP files saved to {foldername}')

        PODP.calc_error_bars_2()
        np.save(foldername + '/error_tensors_PODP.npy', MPS.error_tensors)

    return PODP
    # Calculating full order sols for test positions.
    MPS_full = MultiParamSweep(0.001, 2, {'air': 0, 'ring': 17e5}, mesh)
    MPS_full.frequency_array = np.logspace(2, 4, 3)
    MPS_full.permeability_array = np.asarray([20, 30, 40])
    # Calculating theta0 and theta1 snapshots:
    MPS_full.calc_theta0_snapshots(apply_postprojection=True)
    MPS_full.calc_theta1_snapshots(apply_postprojection=False)
    MPS_full.calc_N0_snapshots()
    MPS_full.calc_R_snapshots()
    MPS_full.calc_I_snapshots()

    # RNN_start_time = time.time()
    print('calculating Eigenvalues')
    eig_full = np.zeros((3, 32, 32), dtype=complex)
    for ind in range(len(MPS_full.permeability_array)):
        for jnd in range(len(MPS_full.frequency_array)):
            eig_full[:, ind, jnd] = np.linalg.eigvals(MPS_full.R[:, :, ind, jnd] + MPS_full.N0_snapshots[ind, :, :]) + 1j* np.linalg.eigvals(MPS_full.I[:, :, ind, jnd])
    foldername = MPS.save_results(suffix='_FULL')
    np.save(foldername + '/eigenvalues.npy', eig_full)


    np.savetxt(foldername + '/timings.csv', [elapsed_time_rnn, elapsed_time_podn, elapsed_time_podp])

    # Producing error estimate:
    print('calculating error')
    error_podp = np.zeros((3, 3))
    error_podn = np.zeros((3, 3))
    error_rnn = np.zeros((3, 3))
    for ind in range(len(MPS_full.permeability_array)):
        for jnd in range(len(MPS_full.frequency_array)):
            l_podp = eig_podp[0, ind, jnd]
            l_full = eig_full[0, ind, jnd]
            l_podn = eig_podn[0, ind, jnd]
            l_rnn = eig_RNN[ind, jnd]
            error_podp[ind, jnd] = np.abs(l_podp - l_full) / np.abs(l_full)
            error_podn[ind, jnd] = np.abs(l_podn - l_full) / np.abs(l_full)
            error_rnn[ind, jnd] = np.abs(l_rnn - l_full) / np.abs(l_full)

    np.savetxt(foldername + '/error_podn.csv', error_podn)
    np.savetxt(foldername + '/error_podp.csv', error_podp)
    np.savetxt(foldername + '/error_rnn.csv', error_rnn)

    return eig_podn, eig_podp, elapsed_time_podn, elapsed_time_podp

# HELPER FUNCTIONS #

def save_MPS(MPS, filename='default'):
    """
    James Elgy - 2022
    Function to save MultiParamSweep class to file.
    :param MPS: Class instance to save.
    :param filename: filename of the saved file. If not 'default' then file name is constructed using the instance
                     attributes.
    :return:
    """
    if filename == 'default':
        tlo_list = [obj for obj in MPS.mesh.GetMaterials() if obj != 'air']
        tlo_list_string = ''.join(tlo_list)
        filename = (
                f'{tlo_list_string}_' +
                f'mur={min(MPS.permeability_array):2.2e}-{max(MPS.permeability_array):2.2e}_' +
                f'omega={min(MPS.frequency_array):2.2e}-{max(MPS.frequency_array):2.2e}_' +
                f'samples={len(MPS.permeability_array)}x{len(MPS.frequency_array)}_' +
                f'alpha={MPS.alpha}' +
                '.pkl'
        )

    with open(filename, 'wb') as f:
        pickle.dump(MPS.__dict__, f)
    print(f'Saved as: {filename}')
    return 0


def load_MPS(filename):
    """
    James Elgy - 2022
    Function to load and populate MultiParamSweep class from file.
    Function operates on a pickle file corresponding to save_MPS().
    :param filename: filename of the pickle file.
    :return: MPS: polulated class.
    """
    dummy_mesh = Mesh("../VolFiles/" + 'sphere.vol')
    MPS = MultiParamSweep(0,0,0, dummy_mesh)
    with open(filename, 'rb') as f:
        MPS.__dict__ = pickle.load(f)
    return MPS


def save_all_figures(path, format='png', suffix='', prefix=''):
    """
    James Elgy - 2021
    Function to save all open figures to disk.
    Files are named as:
    {suffix}{figure_n}{prefix}.{format}
    :param path: path to the desired saving directory.
    :param format: desired file format. pdf, png, jpg, tex, pickle
    :param suffix: additional component of the output filename
    :param prefix: additional component of the output filename
    :return:

    EDIT 04/01/2022:
    Added pickle support to save interactive figures.

    """

    if not os.path.isdir(path):
        os.mkdir(path)
    extension = '.' + format
    if format != 'tex' and format != 'pickle':
        for i in plt.get_fignums():
            plt.figure(i)
            filename = prefix + f'figure_{i}' + suffix
            plt.savefig(os.path.join(path, filename) + extension)
    elif format == 'tex':
        for i in plt.get_fignums():
            plt.figure(i)
            filename = prefix + f'figure_{i}' + suffix
            tikzplotlib.save(os.path.join(path, filename) + extension)
    elif format == 'pickle':
        for i in plt.get_fignums():
            filename = prefix + f'figure_{i}' + suffix
            full_path = os.path.join(path, filename) + extension
            pickle.dump(plt.figure(i), open(full_path, 'wb'))
    else:
        raise TypeError('Unrecognised file format')


def save_figure(path, filename, format='png'):
    """
    James Elgy - 2021
    Function to save the current figure to disk.
    Files are named as:
    {suffix}{figure_n}{prefix}.{format}
    :param path: path to the desired saving directory.
    :param format: desired file format. pdf, png, jpg, tex, pickle
    :param filename: output file name
    :return:

    EDIT 04/01/2022:
    Added pickle support to save interactive figures.

    """

    if not os.path.isdir(path):
        os.mkdir(path)
    extension = '.' + format
    if format != 'tex' and format != 'pickle':
        plt.savefig(os.path.join(path, filename) + extension)
    elif format == 'tex':
        tikzplotlib.save(os.path.join(path, filename) + extension)
    elif format == 'pickle':
        full_path = os.path.join(path, filename) + extension
        pickle.dump(plt.gcf(), open(full_path, 'wb'))
    else:
        raise TypeError('Unrecognised file format')


def load_figure(filenames):
    """
    James Elgy - 2022
    Function to load pickled figures from disk.
    :param path: str path to pickle file
    :return:
    """

    for file in filenames:
        fig = pickle.load(open(file, 'rb'))


if __name__ == '__main__':

    MPS = eval_error_at_sample_points()


    # MPS = main()
    # start_time = time.time()
    # mem_usage = memory_usage(eval_error_at_sample_points())
    # elapsed_time = time.time() - start_time
    # print(f'Total Time: {elapsed_time / 3600} Hours')
    # plt.figure(999)
    # plt.plot(np.linspace(0, 0.2*len(mem_usage), len(mem_usage)), mem_usage, label=label)
    # plt.xlabel('time [sec]')
    # plt.ylabel('Mem Usage [MB]')
