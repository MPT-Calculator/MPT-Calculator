import os
import pickle
import sys
# import tkinter.filedialog  # Used for file dialog in figure loader.
# from tkinter import Tk

import numpy as np
import tikzplotlib  # Used for saving figures as tikz figures.
from scipy.signal import find_peaks  # Used for peak finding.

# sys.path.insert(0, "Functions")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + '/Settings')
from MPTFunctions import *
from matplotlib import pyplot as plt
from sklearn import neural_network as nn
from sklearn import model_selection as ms
from sklearn import preprocessing as pp
import multiprocessing
import psutil
import SingleSolve
import Settings
import netgen.meshing as ngmeshing
import pandas as pd
import scipy.sparse as sp



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
    if format != 'tex':
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


""" NEURAL NETWORK POD """


def POD_NN_permeability(fes, Order, alpha, mu, inout, evec, Tolerance, Maxsteps, epsi, simnumber, Solver, sig,
                        sanity_check_N0=False, allow_direction_dependence=True):
    permeability_array = np.logspace(1, 1, 1)
    frequency_array = np.logspace(1, 1, 1)
    # permeability_array = permeability_array[1:]
    # permeability_array = np.concatenate((np.logspace(-2,-0.01,10), permeability_array))
    N0_curve_exact = []
    N0_curve_approx = []
    if sanity_check_N0 == True:
        D = np.zeros((fes.ndof, len(permeability_array), 3))
        for ind, mu in enumerate(permeability_array):
            for i in range(3):
                mur = {'air': 1.0, 'sphere': mu}
                mur_coeff = CoefficientFunction([mur[mat] for mat in fes.mesh.GetMaterials()])
                D[:, ind, i] = Theta0(fes, Order, alpha, mur_coeff, inout, evec[i], Tolerance, Maxsteps, epsi, i + 1,
                                      Solver)
            Theta0Sol = np.squeeze(D[:, ind, :])
            N0_approx, N0_exact = _sanity_check_N0(Theta0Sol, mur_coeff, fes, mu, alpha)
            diff_1 = N0_approx[0, 0] - N0_exact[0, 0]
            print(
                f'\n\nmur = {mu} \nN0 FEM = {N0_approx[0, 0]} \nN0 exact = {N0_exact[0, 0]} \nN0 difference = {diff_1} \nFrac error = {diff_1 / N0_exact[0, 0]} \n\n')

            N0_curve_exact += [[np.diag(N0_exact)]]
            N0_curve_approx += [[np.diag(N0_approx)]]

        plt.figure()
        plt.plot(permeability_array, N0_curve_exact, marker='x')
        plt.plot(permeability_array, N0_curve_approx, marker='+')

    else:
        if allow_direction_dependence == True:
            N_dim = 3
        else:
            N_dim = 1

        D = np.ones((fes.ndof, len(permeability_array), N_dim))

        for dim in range(N_dim):
            for ind, mu in enumerate(permeability_array):
                mur = {'air': 1.0, 'sphere': mu}
                mur_coeff = CoefficientFunction([mur[mat] for mat in fes.mesh.GetMaterials()])

                # mat_names = fes.mesh.GetMaterials()
                # mur_values = [mu if name != 'air' else 0 for name in mat_names]
                # mur = dict(zip(mat_names, mur_values))
                # mur_coeff = CoefficientFunction([mur[mat] for mat in mat_names])

                # simnumber = 1
                D[:, ind, dim] = Theta0(fes, Order, alpha, mur_coeff, inout, evec[dim], Tolerance, Maxsteps, epsi, dim,
                                        Solver)
                print(f'Solved for permeability {mu}, ind = {ind}, dir={dim}')
            # U_trunc, S_trunc, Vh_trunc = truncated_SVD(D, 1e-3)

        np.save('Theta0Sol_Sphere_0.4_log10_mur=0_to_2_211_samples_al=0.01_sig=1e6_220126', D)
        pickle.dump(fes, open('FES_Sphere_0.4_log10_mur=0_to_2_11_samples_al=0.01_sig=1e6_220126.pkl', 'wb'))

        # Calculating theta1 solutions.
        calc_theta1 = True
        if calc_theta1 == True:

            dom_nrs_metal = [0 if mat == "air" else 1 for mat in fes.mesh.GetMaterials()]
            fes2 = HCurl(fes.mesh, order=Order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
            D1_full = np.zeros((fes2.ndof, 3, len(permeability_array), len(frequency_array)), dtype=complex)

            for ind, mur in enumerate(permeability_array):
                mur = mur
                mur_dict = {'air': 1, 'sphere': mur}
                mur_coeff = [mur_dict[mat] for mat in fes.mesh.GetMaterials()]
                for jnd, omega in enumerate(frequency_array):
                    print(f'frequency {jnd}, permeability {ind}')
                    D1 = POD_NN_theta1(fes, fes2, D[:, ind, :], Order, alpha, sig, inout, Tolerance, Maxsteps, epsi,
                                       omega, Solver, mur_coeff)
                    D1_full[:, :, ind, jnd] = D1

            np.save('Theta0Sol_Sphere_0.2_log10_mur=0_to_2_6x7_samples_al=0.01_sig=1e6_220127', D)
            pickle.dump(fes2, open('FES2_Sphere_0.2_log10_mur=0_to_2_6x7_samples_al=0.01_sig=1e6_220127.pkl', 'wb'))

    sys.exit(0)


def POD_NN_theta1(fes, fes2, Theta0Sol, order, alpha, sigma_coeff, inout, tolerance, maxsteps, epsi, omega, Solver, mur_coeff):
    xivec = [CoefficientFunction((0, -z, y)), CoefficientFunction((z, 0, -x)), CoefficientFunction((-y, x, 0))]
    Theta1Sol = np.zeros([fes2.ndof, 3], dtype=complex)
    # mu = CoefficientFunction(mur_coeff)
    nu = omega * 4 * np.pi * 1e-7 * (alpha ** 2)
    # sig = {'air': 0, 'sphere': conductivity}
    # sigma_coeff = [sig[mat] for mat in fes.mesh.GetMaterials()]
    # sigma = CoefficientFunction(sigma_coeff)

    for i in range(3):
        Theta1Sol[:, i] = Theta1(fes, fes2, Theta0Sol[:, i], xivec[i], order, alpha, nu, sigma_coeff, mur_coeff, inout, tolerance,
                                 maxsteps, epsi, omega, i + 1,
                                 3, Solver)

    return Theta1Sol


def POD_NN_multithreaded(fes, Order, alpha, mu, inout, evec, Tolerance, Maxsteps, epsi, simnumber, Solver, sig):
    """
    James Elgy - 2022
    This is a parallel version of the POD_NN_permeability function.
    The function calculates the theta0 and theta1 solutions for a range of permeabilities and frequencies.
    Theta0 is stored as a real ndof x n_perm x 3 array, theta1 is stored as complex ndof x 3 x n_perm x n_freq array

    Function saves theta0 and theta1 solutions to disk, along with the respective fes-es. Currently the filenames are
    hardcoded.

    :param fes: Finite element space for the theta0 problem.
    :param Order: order of the finite element basis functions.
    :param alpha: object size in m i.e. 0.01 m.
    :param mu: Currently unused. Will probably be introduced if the perm and freq loops move outside the function.
    :param inout: Binary dictionary of if the top level objects in the geo file are considered to be part of the object.
    :param evec: coefficient function for the underlying coordinate system. ie. evec = [ CoefficientFunction( (1,0,0) ), CoefficientFunction( (0,1,0) ), CoefficientFunction( (0,0,1) ) ]
    :param sig: dictionary of conductivity values. {'air':0, 'objname':sig}

    Solver Settings:
    :param Tolerance: Conjugate gradient solver tolerance.
    :param Maxsteps: Max number of iterations.
    :param epsi: Regularisation term
    :param simnumber: Currently unused. Will keep track of which dimension is being solved for.
    :param Solver: Solver preconditioner. I.e. "bddc"

    :return:
    """



    # For testing: adding overwrites input parameters
    Solver, epsi, Maxsteps, Tolerance = Settings.SolverParameters()
    ngmesh = ngmeshing.Mesh(dim=3)
    ngmesh.Load("VolFiles/" + 'sphere.vol')
    mesh = Mesh("VolFiles/" + 'sphere.vol')
    mesh.Curve(5)
    alpha = 0.01
    Order = 3
    inorout = {'air': 0, 'sphere': 1}
    inout_coef = [inorout[mat] for mat in mesh.GetMaterials()]
    inout = CoefficientFunction(inout_coef)
    sig = {'air': 0.0, 'sphere': 1e6}
    fes = HCurl(mesh, order=Order, dirichlet="outer", flags={"nograds": True})
    evec = [ CoefficientFunction( (1,0,0) ), CoefficientFunction( (0,1,0) ), CoefficientFunction( (0,0,1) ) ]








    # Log spacing produces a faster decaying SVD.
    permeability_array = np.logspace(0, 2, 18)
    permeability_array = np.linspace(1.5,1.5,1)
    frequency_array = np.logspace(1, 8, 18)

    # Theta0 Problem: ______________________________________________________________________________________
    print('Solving Theta0 Problem')
    D0_full = np.zeros((fes.ndof, len(permeability_array), 3))

    for dim in [0, 1, 2]: #  For each dimension in evec calculate theta0 and store.
        input = []
        for ind, mu in enumerate(permeability_array):
            mur = {'air': 1.0, 'sphere': mu}
            mur_coeff = CoefficientFunction([mur[mat] for mat in fes.mesh.GetMaterials()])
            new_input = (fes, Order, alpha, mur_coeff, inout, evec[dim], Tolerance, Maxsteps, epsi, dim,
                         Solver)
            input.append(new_input)

        with multiprocessing.Pool(processes=3, initializer=_set_niceness) as pool: #  _set_niceness is used to reduce the priority of the process so that it does not slow down the pc.
            output = pool.starmap(Theta0, input)
        print(output)

        for ind, mu in enumerate(permeability_array):
            D0_full[:, ind, dim] = output[ind]

    np.save('Theta0Sol_Test', D0_full)
    pickle.dump(fes, open('FES_Test.pkl', 'wb'))

    # Calculating N0 tensor. Stored as n_perm x 3 x 3 array.
    N0, _ = calc_N0_from_D(D0_full, fes, alpha, permeability_array)
    N0 = np.asarray(N0)

    # Applying postprocessing: _____________________________________________________________________________
    proj = _define_postprojection(fes)

    theta0 = GridFunction(fes)
    for dim in range(3):
        for ind in range(len(permeability_array)):
            theta0.vec.FV().NumPy()[:] = D0_full[:,ind, dim]
            theta0.vec.data = proj * (theta0.vec)
            D0_full[:, ind, dim] = theta0.vec.FV().NumPy()[:]

    # Theta1 problem: ______________________________________________________________________________________
    print('Solving Theta1 Problem')

    dom_nrs_metal = [0 if mat == "air" else 1 for mat in fes.mesh.GetMaterials()]
    fes2 = HCurl(fes.mesh, order=Order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    D1_full = np.zeros((fes2.ndof, 3, len(permeability_array), len(frequency_array)), dtype=complex)
    input = []
    for ind, mu in enumerate(permeability_array):
        for jnd, omega in enumerate(frequency_array):
            mur = {'air': 1.0, 'sphere': mu}
            mur_coeff = CoefficientFunction([mur[mat] for mat in fes2.mesh.GetMaterials()])
            sig_coeff = CoefficientFunction([sig[mat] for mat in fes2.mesh.GetMaterials()])
            new_input = (
            fes, fes2, D0_full[:, ind, :], Order, alpha, sig_coeff, inout, Tolerance, Maxsteps, epsi, omega, Solver,
            mur_coeff)
            input.append(new_input)

    with multiprocessing.Pool(processes=5, initializer=_set_niceness) as pool:
        output = pool.starmap(POD_NN_theta1, input)
    print(output)

    output = np.asarray(output)
    output = np.reshape(output, (len(frequency_array), len(permeability_array), fes2.ndof, 3))

    for ind, mu in enumerate(permeability_array):
        for jnd, omega in enumerate(frequency_array):
            for dim in range(3):
                D1_full[:, dim, ind, jnd] = output[jnd, ind, :,dim]

    np.save('Theta1Sol_Test', D1_full)
    pickle.dump(fes2, open('FES2_Test.pkl', 'wb'))

    # Calculating MPT coefficients.
    print('Calculating MPT coeffs')
    R = np.empty((len(permeability_array), len(frequency_array), 3, 3))
    I = np.empty((len(permeability_array), len(frequency_array), 3, 3))
    MPT = np.empty((len(permeability_array), len(frequency_array), 3, 3), dtype=complex)
    xivec = [CoefficientFunction((0, -z, y)), CoefficientFunction((z, 0, -x)), CoefficientFunction((-y, x, 0))]

    for ind, mu in enumerate(permeability_array):
        mur = {'air': 1.0, 'sphere': mu}
        mur_coeff = CoefficientFunction([mur[mat] for mat in fes2.mesh.GetMaterials()])
        for jnd, omega in enumerate(frequency_array):
            nu = omega * 4 * np.pi * 1e-7 * (alpha ** 2)
            c1, c2 = calc_MPT_coeffs(fes, fes2, D1_full[:,:,ind, jnd], D0_full[:,ind,:],xivec,alpha,mur_coeff,sig_coeff,inout,nu )
            R[ind,jnd, :, :] = c1
            I[ind,jnd, :, :] = c2

            MPT[ind,jnd, :, :] = N0[ind,:,:] + R[ind,jnd,:,:] + 1j*I[ind,jnd,:,:]

    np.save('MPT_Sphere_0.4_log10_mur=0_to_2_freq=1_to_8_18x18_samples_al=0.01_sig=1e6_220214', MPT)


    sys.exit(0)


def truncated_SVD(D, tol, n_elements=None):
    """
    James Elgy - 2021
    Perform truncated singular value decomposition of matrix D approx U_trunc @ S_trunc @ Vh_trunc.
    :param D: NxP matrix
    :param tol: float tolerance for the truncation
    :param n_elements: int override for the tolerance parameter to force a specific number of singular values.
    :return: U_trunc, S_trunc, Vh_trunc - Truncated left singular matrix, truncated singular values, and truncated
    hermitian of right singular matrix.
    """

    U, S, Vh = np.linalg.svd(D, full_matrices=False)  # Zeros are removed.
    S_norm = S / S[0]
    for ind, val in enumerate(S_norm):
        if val < tol:
            break
        cutoff_index = ind
    if n_elements != None:
        cutoff_index = n_elements - 1

    S_trunc = np.diag(S[:cutoff_index + 1])
    U_trunc = U[:, :cutoff_index + 1]
    Vh_trunc = Vh[:cutoff_index + 1, :]

    return U_trunc, S_trunc, Vh_trunc


def neural_network_1D(x_array, y_array, x_query, neurons=(32, 32), activation='logistic', tol=1e-10, alpha=0):
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
    x_scaled = x_scaler.fit_transform(x_array)
    y_scaler = pp.StandardScaler()
    y_scaled = y_scaler.fit_transform(y_array)

    # Building and training network.
    regressor = nn.MLPRegressor(hidden_layer_sizes=neurons, max_iter=5000000, activation=activation,
                                solver='lbfgs', tol=tol, alpha=alpha, verbose=False, warm_start=True,
                                random_state=None, n_iter_no_change=1000, max_fun=1000000)

    regression = regressor.fit(x_scaled, np.ravel(y_scaled))

    # Making prediction
    scaled_query = (x_scaler.transform(x_query))
    y_pred = regressor.predict(scaled_query)
    y_pred = y_scaler.inverse_transform(y_pred)  # Denormalising.

    return y_pred, regressor


def neural_network_2D(x_array, y_array, z_array, x_query, y_query, neurons=(32,32), activation='tanh', tol=1e-10, alpha=0.00001, scaler='standard'):
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
    if scaler == 'standard':
    # Normalising input data to mean=0, std=1.
        x_scaler = pp.StandardScaler()
        y_scaler = pp.StandardScaler()
        z_scaler = pp.StandardScaler()
    else:
        x_scaler = pp.MinMaxScaler()
        y_scaler = pp.MinMaxScaler()
        z_scaler = pp.MinMaxScaler()

    x_scaled = x_scaler.fit_transform(x_array)
    y_scaled = y_scaler.fit_transform(y_array)
    z_scaled = z_scaler.fit_transform(z_array)

    regressor = nn.MLPRegressor(hidden_layer_sizes=neurons, max_iter=500000, activation=activation,
                                solver='lbfgs', tol=tol, alpha=alpha, verbose=False, warm_start=False,
                                random_state=None, n_iter_no_change=10000, max_fun=100000)


    # constucting input array and training model
    total_samples = len(x_array)*len(y_array)
    xx,yy = np.meshgrid(x_scaled, y_scaled)
    input_array = np.asarray([np.ravel(xx, order='F'), np.ravel(yy, order='F')]).transpose()
    regression = regressor.fit(input_array, z_scaled.flatten())

    # Making prediction
    scaled_query_x = (x_scaler.transform(x_query))
    scaled_query_y = (y_scaler.transform(y_query))

    xxq, yyq = np.meshgrid(scaled_query_x, scaled_query_y)
    query_array = np.asarray([np.ravel(xxq, order='F'), np.ravel(yyq, order='F')]).transpose()
    scaled_query_z = regression.predict(query_array)

    # Undoing normalisation
    if scaler != 'standard':
        z_query = z_scaler.inverse_transform(scaled_query_z.reshape((len(query_array),1)))
    else:
        z_query = z_scaler.inverse_transform(scaled_query_z)

    plt.figure()
    plt.plot(z_query, label='query')
    plt.plot(np.ravel(z_array), label='input')
    plt.legend()
    plt.xlabel('Snapshot Number')
    plt.ylabel('$V^H$ mode 1')

    return np.squeeze(z_query)


def eval_network_performance_1D(x_array, y_array, test_proportion=0.1, plot_figure=True):
    """
    James Elgy - 2022
    Function to evaluate the normalised RMSE associated with the 1D fitting performed using the neural network.
    :param x_array: np array of the total set of x values
    :param y_array: np array of the total set of y values
    :param test_proportion: float 0<p<1 that defined the proportion of the total sets to be used for testing.
    :return: NRMSE The normalised root mean squared error.
    """

    x_train, y_train, x_test, y_test = _train_test_split(x_array, y_array, test_proportion, plot_figure=False)
    y_pred, regression = neural_network_1D(x_train.reshape((len(x_train), 1)), y_train.reshape((len(y_train), 1)),
                                           x_test.reshape((len(x_test), 1)))
    MSE = np.sum((y_pred - np.squeeze(y_test)) ** 2) / len(y_test)  # squeeze ensures same shape.
    RMSE = np.sqrt(MSE)
    NRMSE = RMSE / (np.sqrt(np.sum(y_test ** 2) / len(y_test)))

    if plot_figure == True:
        plt.figure()
        plt.scatter(x_train, y_train, color='r', label='training')
        plt.scatter(x_test, y_test, color='b', label='testing')

        x_plot_array = np.linspace(np.min(x_array), np.max(x_array), 200)
        y_plot_array, _ = neural_network_1D(x_train.reshape((len(x_train), 1)), y_train.reshape((len(y_train), 1)),
                                            x_plot_array.reshape((len(x_plot_array), 1)))
        plt.plot(x_plot_array, y_plot_array, color='k', label='NN prediction')
        plt.legend()
        plt.xlabel('relative permeability')
        plt.ylabel('$V^H$')

    return NRMSE


def POD_NN_permeability_from_disk(filename, permeability_array, svd_tol=1e-3, N_modes=None, projection=None):
    """
    James Elgy - 2022
    Loads Theta0Sol matrix from disk and performs SVD. The function then takes the first 2 modes of the right singular
    matrix and performs neural network based interpolation. The interpolated right singular matrix is then premultiplied
    to produces an interpolated estimate of the original Theta0Sol matrix.
    :param filename:
    :param alpha:
    :return:
    """

    # D_full = np.genfromtxt(filename, delimiter=',', dtype='float')
    D_full = np.load(filename)
    D_full = D_full[:, 1:, :]  # Removing mur=1 from the underlying array.

    # permeability_array = np.logspace(0, 2, 21)
    # permeability_array = permeability_array[1:]

    # permeability_array = np.logspace(0, 2, 50)
    # permeability_array = np.concatenate((np.logspace(-2, -0.01, 10), permeability_array))
    # permeability_array = permeability_array[1:]

    D_trunc = np.zeros(D_full.shape)
    D_pred = np.zeros(D_full.shape)
    plt.figure(1)
    for dim in range(3):
        D = D_full[:, :, dim]
        n_snapshots = len(permeability_array)
        if type(projection) == tuple:
            theta0 = GridFunction(projection[1])
            for ind in range(n_snapshots):
                theta0.vec.FV().NumPy()[:] = D[:, ind]
                theta0.vec.data = projection[0] * (theta0.vec)
                D[:, ind] = theta0.vec.FV().NumPy()[:]

        if N_modes == None:
            U_trunc, S_trunc, Vh_trunc = truncated_SVD(D, svd_tol)
        else:
            U_trunc, S_trunc, Vh_trunc = truncated_SVD(D, svd_tol, n_elements=N_modes)

        n_modes = len(np.diag(S_trunc))
        print(f'N modes = {n_modes}')
        plt.figure(1)
        plt.semilogy(np.diag(S_trunc) / S_trunc[0, 0], marker='x', label=f'Singular Values : dim = {dim + 1}')
        plt.xlabel('column index')
        plt.ylabel('Normalised Singular Values')
        plt.legend()

        plt.figure()
        for ind in range(n_modes):
            plt.semilogx(permeability_array, Vh_trunc[ind, :], marker='x', label=f'mode = {ind}' + f' dim={dim + 1}')
        plt.legend()
        plt.xlabel('Relative Permeability')
        plt.ylabel('$V^H_m$')

        # Applying Neural Network for all retained modes.
        query_permeabilities = np.log10(permeability_array)
        Vh_pred = np.zeros((n_modes, n_snapshots))
        for mode in range(n_modes):
            y_array = Vh_trunc[mode, :]

            # Checking for delta function. If True, does not train network on that mode.
            flag, peak_pos = _check_delta_function(y_array, peakheight=0.95, mean_tol=1e-2)
            if flag == False:
                X_train, X_test, y_train, y_test = ms.train_test_split(permeability_array, y_array, test_size=0.2, random_state=42)

                y_pred, _ = neural_network_1D(np.log10(X_train.reshape((len(X_train), 1))),
                                              y_train.reshape((len(y_train), 1)),
                                              query_permeabilities.reshape((n_snapshots, 1)))
                plt.plot(10 ** query_permeabilities, y_pred, label=f'NN prediction {mode}' + f' dim={dim}')
                Vh_pred[mode, :] = y_pred
            else:
                y_pred = [y_array[ind] if ind == peak_pos else 0 for ind in range(n_snapshots)]
                Vh_pred[mode, :] = y_pred

        plt.legend()

        # Calculating Theta0Sol matrix and approximating errors.
        D_pred[:, :, dim] = U_trunc @ S_trunc @ Vh_pred
        D_trunc[:, :, dim] = U_trunc @ S_trunc @ Vh_trunc

    return D_trunc, D_pred, D_full


def calc_N0_from_D(D, fes, alpha, permeability_array):
    """
    James Elgy - 2022
    Function to calculate N0 from Theta0 solution vectors for a given size, fes, and permeability.
    Currently only for spheres.
    :param D: NxP matrix of solution vectors.
    :param fes: NGSolve finite element space for the object.
    :param alpha: float object size (m)
    :param permeability_array: P relative permeabilities.
    :return: N0_array, N0_exact_array - mx3x3 arrays of the 3x3 N0 and 3x3 exact solution for N0 for each permeability.
    """

    mesh = fes.mesh
    N0_array = []
    N0_exact_array = []

    for ind, mu in enumerate(permeability_array):  # Assigning permeability
        print(f'Calculating N0 - mur = {mu} - ind = {ind}')

        Theta0i = GridFunction(fes)
        Theta0j = GridFunction(fes)
        N0 = np.zeros((3, 3))

        mur = {'air': 1.0, 'sphere': mu}
        # mat_names = fes.mesh.GetMaterials()
        # mur_values = [mu if name != 'air' else 0 for name in mat_names]
        # mur = dict(zip(mat_names, mur_values))
        mur_coeff = CoefficientFunction([mur[mat] for mat in fes.mesh.GetMaterials()])
        # Theta0Sol = np.tile(D[:,ind],(3,1)).transpose()  # For sphere all directions give same result.
        Theta0Sol = D[:, ind, :]

        # Calculate the N0 tensor
        VolConstant = Integrate(1 - mur_coeff ** (-1), mesh)
        for i in range(3):
            Theta0i.vec.FV().NumPy()[:] = Theta0Sol[:, i]
            for j in range(3):
                Theta0j.vec.FV().NumPy()[:] = Theta0Sol[:, j]
                if i == j:
                    N0[i, j] = (alpha ** 3) * (VolConstant + (1 / 4) * (
                        Integrate(mur_coeff ** (-1) * (InnerProduct(curl(Theta0i), curl(Theta0j))), mesh)))
                else:
                    N0[i, j] = (alpha ** 3 / 4) * (
                        Integrate(mur_coeff ** (-1) * (InnerProduct(curl(Theta0i), curl(Theta0j))), mesh))

        # Copy the tensor
        N0 += np.transpose(N0 - np.eye(3) @ N0)
        N0_array += [N0]

        # Calc exact N0
        N0_exact = [2 * np.pi * alpha ** 3 * (
                2 * (mu - 1) / (mu + 2))] * 3  # converting N0 from float to 3x3 multiple of identity
        N0_exact = np.diag(N0_exact)
        N0_exact_array += [N0_exact]

    return N0_array, N0_exact_array


def calc_MPT_coeffs(fes,fes2,Theta1Sol,Theta0Sol,xivec,alpha,mu,sig_coeff,inout,nu, fast_comp=False):
    mesh = fes.mesh

    if fast_comp == False:
        R , I = MPTCalculator(mesh, fes, fes2, Theta1Sol[:,0], Theta1Sol[:,1], Theta1Sol[:,2], Theta0Sol, xivec, alpha,mu,sig_coeff,inout,nu, "No Print",1)
    else:

        # Real part:
        u, v = fes.TnT()
        N = GridFunction(fes)
        for i in range(3):
            for j in range(3):
                rij = BilinearForm(fes)
                rij += inout * mu**(-1) * curl(v[i]) * Conj(curl(v[j])) * dx
                rij += -(inout-1) * curl(v[i]) * Conj(curl(v[j])) * dx  # inout-1 is negative for outside object.
                rij.Assemble()


    return R, I


""" HELPER FUNCTIONS """


def _sanity_check_N0(Theta0Sol, mu, fes, mur, alpha):
    """
    James Elgy - 2021
    In the case of a sphere calcs N0(approx) and N0(exact) for the relative permeability in question. Should show good
    agreement.

    :param Theta0Sol: Ndof x 3 solutions to the theta0 problem.
    :param mu: Ngsolve CoefficientFunction for relative permeability {air:0, sphere:mur} <- different for each material.
    :param fes: Finite element space.
    :param mur: float relative permeability of the object.
    :param alpha: scaling term for object size.
    :return: N0, N0_exact - approximation and exact solution for N0.
    """

    # Preallocation
    Theta0i = GridFunction(fes)
    Theta0j = GridFunction(fes)
    N0 = np.zeros((3, 3))
    mesh = fes.mesh

    # Calculate the N0 tensor
    VolConstant = Integrate(1 - mu ** (-1), mesh)
    for i in range(3):
        Theta0i.vec.FV().NumPy()[:] = Theta0Sol[:, i]
        for j in range(3):
            Theta0j.vec.FV().NumPy()[:] = Theta0Sol[:, j]
            if i == j:
                N0[i, j] = (alpha ** 3) * (VolConstant + (1 / 4) * (
                    Integrate(mu ** (-1) * (InnerProduct(curl(Theta0i), curl(Theta0j))), mesh)))
            else:
                N0[i, j] = (alpha ** 3 / 4) * (
                    Integrate(mu ** (-1) * (InnerProduct(curl(Theta0i), curl(Theta0j))), mesh))

    # Copy the tensor
    N0 += np.transpose(N0 - np.eye(3) @ N0)

    # Calc exact N0
    N0_exact = [2 * np.pi * alpha ** 3 * (
            2 * (mur - 1) / (mur + 2))] * 3  # converting N0 from float to 3x3 multiple of identity
    N0_exact = np.diag(N0_exact)

    return N0, N0_exact


def _sanity_check_neural_network_1d():
    """
    James Elgy - 2022
    Function to generate box plot and test performance of neural network.
    :return:
    """

    x = np.linspace(-1, 1, 100)
    y = np.asarray([1 if -0.5 < element < 0.5 else 0 for element in x])
    plt.figure()
    plt.scatter(x, y, color='b', label='input data')

    x_query = np.linspace(-1, 1, 200)
    y_pred, _ = neural_network_1D(x.reshape((100, 1)), y.reshape((100, 1)), x_query.reshape((200, 1)))
    plt.plot(x_query, y_pred, color='r', label='NN prediction')
    plt.legend()


def _train_test_split(x_array, y_array, proportion, plot_figure=True):
    """
    James Elgy - 2022
    Function to split input array into training and testing subsets. Helper function to be used in neural_network_1D.
    :param x_array: np array of the x values for the total set.
    :param y_array: np array of the y values for the total set.
    :param proportion: float proportion of the total set to be used as the test set.
    :return: x_train, y_train, x_test, y_test. The training and testing subsets.
    """
    n_elements = len(x_array)
    n_test_elements = int(np.floor(n_elements * proportion))
    n_training_elements = n_elements - n_test_elements

    test_indices = np.random.permutation(n_elements)[:n_test_elements]
    x_test = x_array[test_indices]
    y_test = y_array[test_indices]
    x_train = np.delete(x_array, test_indices)  # Delete does not occur in-place.
    y_train = np.delete(y_array, test_indices)

    if plot_figure == True:
        plt.figure()
        plt.scatter(x_train, y_train, color='r', label='training')
        plt.scatter(x_test, y_test, color='b', label='testing')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

    return x_train, y_train, x_test, y_test


def _check_delta_function(y_array, peakheight=0.9, mean_tol=1e-3, std_tol=1e-3):
    """
    James Elgy - 2022
    Function to check if the input distribution is a delta function. Function checks for one entry above tolerance and
    if other values are approximately 0. Returns index of the peak.

    :param y_array: numpy array in question
    :param peakheight: minimum peak height between 0 and 1.
    :param mean_tol: Tolerance in the mean of the normalised distribution.
    :param std_tol: Tolerance in the standard deviation of the normalised distribution.
    :return: flag, peak_index: bool (true if delta function) and the index of the peak.
    """

    # To account for a negative delta function, we convert to absolute values and normalise.
    y = np.abs(y_array)
    y = y - np.min(y) / (np.max(y) - np.min(y))

    y = np.concatenate(([0], y, [0]))  # padding array to allow find_peaks to detect peaks in the edges of y_array.

    # Finding peaks.
    peaks, _ = find_peaks(y, height=peakheight)
    if len(peaks) > 1 or len(peaks) == 0:
        return False, 0
    else:
        peak = int(peaks)
        std = np.std(y[np.arange(len(y)) != peak])
        print(f'Delta check Std = {std}')
        mean = np.mean(y[np.arange(len(y)) != peak])
        print(f'Delta check Mean = {mean}')
        print(f'\n\n')

        if (mean > mean_tol) or (std > std_tol):
            return False, 0
        else:
            return True, int(peak - 1)


def _generate_test_delta_function():
    x = np.linspace(0, 100, 101)
    p = 0
    sigma = 0.0005
    a = 1
    b = 0

    n = sigma * np.random.randn(len(x))
    y = [a + b if ind == p else b for ind in x] + n

    plt.figure()
    plt.plot(y, marker='x')

    return y


def _load_fes_from_disk(filename):
    """
    James Elgy - 2022
    Function to load ngsolve fes from pickle file.
    :param filename: path to pickle file
    :return: fes - ngsolve finite element space.
    """

    with open(filename, 'rb') as inp:
        fes = pickle.load(inp)
    return fes


def _load_D_from_disk(filename):
    """
    James Elgy 2022
    Function to load theta0 solutions from npy file.
    :param filename:
    :return:
    """

    D = np.load(filename)
    return D


def _eval_nn_hyperparameters():
    N_neurons = np.logspace(0, 8, 9, base=2, dtype=int)
    N_layers = np.linspace(1, 10, 10, dtype=int)
    xx, yy = np.meshgrid(N_layers, N_neurons)

    performance_matrix = np.zeros((10, 9))
    performance_matrix_N0 = np.zeros((10, 9))

    filename = r'C:\Users\James\Desktop\MPT_calculator_James\Theta0Sol_sphere_0.08_log10_mur=0_to_2_21_samples_al=0.01_sig=1e6_220116.npy'
    fes = _load_fes_from_disk(
        r'C:\Users\James\Desktop\MPT_calculator_James\FES_sphere_0.08_log10_mur=0_to_2_21_samples_al=0.01_sig=1e6_220116.pkl')
    D_full = _load_D_from_disk(filename)
    D_full = D_full[:, 1:, :]
    D_pred = np.zeros(D_full.shape)
    permeability_array = np.logspace(0, 2, 21)
    permeability_array = permeability_array[1:]

    for ind, layers in enumerate(N_layers):
        for jnd, neurons in enumerate(N_neurons):
            D_pred = np.zeros(D_full.shape)
            for dim in [0]:
                U_trunc, S_trunc, Vh_trunc = truncated_SVD(D_full[:, :, dim], 1e-3, n_elements=8)
                n_modes = len(np.diag(S_trunc))
                n_snapshots = len(permeability_array)

                Vh_pred = np.empty((n_modes, n_snapshots))
                for mode in range(n_modes):
                    query_permeabilities = permeability_array
                    y_array = Vh_trunc[mode, :]

                    # Checking for delta function. If True, does not train network on that mode.
                    flag, peak_pos = _check_delta_function(y_array, peakheight=0.95, mean_tol=1e-2)
                    if flag == False:
                        y_pred, _ = neural_network_1D(permeability_array.reshape((n_snapshots, 1)),
                                                      y_array.reshape((n_snapshots, 1)),
                                                      query_permeabilities.reshape((n_snapshots, 1)),
                                                      neurons=(neurons,) * layers)
                        Vh_pred[mode, :] = y_pred
                    else:
                        y_pred = [y_array[ind] if ind == peak_pos else 0 for ind in range(n_snapshots)]
                        Vh_pred[mode, :] = y_pred
                D_pred[:, :, dim] = U_trunc @ S_trunc @ Vh_pred

            for dim in [0]:
                frac_norm = np.linalg.norm(D_full[:, :, dim] - D_pred[:, :, dim]) / np.linalg.norm(D_full[:, :, dim])
                performance_matrix[ind, jnd] = frac_norm

            N0_approx, _ = calc_N0_from_D(D_pred, fes, 0.01, permeability_array)
            N0_full, _ = calc_N0_from_D(D_full, fes, 0.01, permeability_array)

            N0_approx = np.asarray(N0_approx)
            N0_full = np.asarray(N0_full)

            frac_norm_N0 = np.linalg.norm(N0_full - N0_approx) / np.linalg.norm(N0_full)
            performance_matrix_N0[ind, jnd] = frac_norm_N0

    plt.figure()
    plt.pcolor(performance_matrix)
    cbar = plt.colorbar()
    cbar.ax.set_label('$||D-D_{reduced}||_F / ||D||_F$')
    plt.xlabel('$log_2$ N neurons')
    plt.ylabel('N layers -1')

    plt.figure()
    plt.pcolor(performance_matrix)
    cbar = plt.colorbar()
    cbar.ax.set_label('$||\mathcal{N}^0-\mathcal{N}^0_{reduced}||_F / ||\mathcal{N}^0||_F$')
    plt.xlabel('$log_2$ N neurons')
    plt.ylabel('N layers -1')

    return performance_matrix, performance_matrix_N0


def _plot_theta0_sols(proj_mask=False, fig=None, plot_curl=False):
    if fig != None:
        plt.figure(fig)
    else:
        plt.figure()

    filename = r'C:\Users\James\Desktop\MPT_calculator_James\Theta0Sol_Sphere_0.08_log10_mur=0_to_2_51_samples_al=0.01_sig=1e6_220124.npy'
    FES_filename = r'C:\Users\James\Desktop\MPT_calculator_James\FES_Sphere_0.08_log10_mur=0_to_2_51_samples_al=0.01_sig=1e6_220124.pkl'
    D_full = _load_D_from_disk(filename)
    fes = _load_fes_from_disk(FES_filename)

    legendstr = f'postprocessing = {proj_mask}'

    theta0 = GridFunction(fes)
    proj = _define_postprojection(fes)

    permeability_array = np.logspace(0, 2, 51)
    L2_norm = np.empty(permeability_array.shape)

    for dim in range(3):
        D = D_full[:, :, dim]
        for ind, mur in enumerate(permeability_array):
            theta0.vec.FV().NumPy()[:] = D[:, ind]

            # plot_curl = False
            if plot_curl == True:
                if proj_mask == True:
                    theta0.vec.data = proj * (theta0.vec)
                integrand = InnerProduct(curl(theta0), curl(theta0))
            else:
                if proj_mask == True:
                    theta0.vec.data = proj * (theta0.vec)
                integrand = InnerProduct(theta0, theta0)
            L2_norm[ind] = Integrate(integrand, fes.mesh) ** 0.5

        plt.loglog(permeability_array, L2_norm, marker='x', label=legendstr + f', $i=${dim}')
        plt.xlabel('$\mu_r$')
        if plot_curl == False:
            plt.ylabel('$||\\theta^{(0)}_i||_{L^2(\Omega)}$')
        else:
            plt.ylabel('$||\\nabla \\times \\theta^{(0)}_i||_{L^2(\Omega)}$')

        plt.legend(loc='upper right')


def _define_postprojection(fes):
    """
    James Elgy - 2022
    Function to calculate the postprocessing mask required to remove the low order gradient terms from the theta0
    solutions. The function follows the example provided  in [1 pg 142-144].

    [1] S. Zaglmayr, “High Order Finite Element Methods for Electromagnetic Field Computation,”
    Johannes Kepler University, 2006.
    :param fes: Hcurl conforming finite element space.
    :return proj: Postprocessing mask.
    """

    u, v = fes.TnT()
    m = BilinearForm(fes)
    m += u * v * dx
    m.Assemble()

    # build gradient matrix as sparse matrix (and corresponding scalar FESpace)
    gradmat, fesh1 = fes.CreateGradient()

    gradmattrans = gradmat.CreateTranspose()  # transpose sparse matrix
    math1 = gradmattrans @ m.mat @ gradmat  # multiply matrices
    math1[0, 0] += 1  # fix the 1-dim kernel
    invh1 = math1.Inverse(inverse="sparsecholesky")

    # build the Poisson projector with operator Algebra:
    proj = IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ m.mat

    return proj


def _svd_comparison():
    filename = r'/home/james/Desktop/MPT_calculator_James/Theta0Sol_Sphere_0.2_log10_mur=0_to_2_freq=1_to_8_25x26_samples_al=0.01_sig=1e6_220209.npy'
    permeability_array = np.logspace(0, 2, 25)
    permeability_array = permeability_array[1:]
    fes = _load_fes_from_disk(
        r'/home/james/Desktop/MPT_calculator_James/FES_Sphere_0.2_log10_mur=0_to_2_freq=1_to_8_25x26_samples_al=0.01_sig=1e6_220209.pkl')
    proj = _define_postprojection(fes)
    D_trunc, D_pred, D1 = POD_NN_permeability_from_disk(filename, permeability_array, svd_tol=1e-10,
                                                        projection=(proj, fes))
    D_trunc, D_pred, D2 = POD_NN_permeability_from_disk(filename, permeability_array, svd_tol=1e-10)

    plt.figure(999)
    for dim in range(3):
        U1, S1, Vh1 = truncated_SVD(D1[:, :, dim], 1e-10, n_elements=8)
        U2, S2, Vh2 = truncated_SVD(D2[:, :, dim], 1e-10, n_elements=8)
        plt.figure(999)
        plt.semilogy(np.diag(S1) / S1[0, 0], marker='x', label=f'postprocessed - dir={dim}')
        plt.semilogy(np.diag(S2) / S2[0, 0], marker='x', label=f'original - dir={dim}')
        plt.legend(loc='lower left')
        plt.xlabel('mode')
        plt.ylabel('Normalised Singular Values')


def _set_niceness():
    """
    James Elgy 2021
    Function to be called as the initializer for multiprocessing. The function sets the process niceness to the
    lowest priority. This means that although cpu usage may be high running the code should not effect normal use of
    the computer
    :return:
    """

    # is called at every process start
    p = psutil.Process(os.getpid())
    if sys.platform == 'win32':
        # set to lowest priority, this is windows only, on Unix use ps.nice(19)
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        p.nice(19)


""" MAIN FUNCTION """


def main_theta0():
    filename = r'C:\Users\James\Desktop\MPT_calculator_James\Theta0Sol_Sphere_0.2_log10_mur=0_to_2_freq=1_to_8_25x26_samples_al=0.01_sig=1e6_220209.npy'
    permeability_array = np.logspace(0, 2, 25)
    permeability_array = permeability_array[1:]
    fes = _load_fes_from_disk(
        r'C:\Users\James\Desktop\MPT_calculator_James\FES_Sphere_0.2_log10_mur=0_to_2_freq=1_to_8_25x26_samples_al=0.01_sig=1e6_220209.pkl')
    proj = _define_postprojection(fes)
    D_trunc, D_pred, D = POD_NN_permeability_from_disk(filename, permeability_array, svd_tol=1e-10, N_modes=7, projection=(proj, fes))
    D_trunc, D_pred_noproj, D = POD_NN_permeability_from_disk(filename, permeability_array, svd_tol=1e-10, N_modes=7)

    # permeability_array = np.logspace(0, 2, 50)
    # permeability_array = np.concatenate((np.logspace(-2, -0.01, 10), permeability_array))

    # permeability_array = permeability_array[1:]
    # permeability_array = np.delete(permeability_array, 10)
    # D_pred = np.delete(D_pred, 10, axis=1)
    # D = np.delete(D, 10, axis=1)

    N0, _ = calc_N0_from_D(D_pred, fes, 0.01, permeability_array)
    N0_exact, _ = calc_N0_from_D(D, fes, 0.01, permeability_array)
    N0_noproj, _ = calc_N0_from_D(D_pred_noproj, fes, 0.01, permeability_array)
    N0 = np.asarray(N0)
    N0_exact = np.asarray(N0_exact)
    N0_noproj = np.asarray(N0_noproj)

    plt.figure()
    plt.semilogx(permeability_array, N0[:, 0, 0], label='postprocessed $\mathcal{N}^0[0,0]$', marker='x')
    plt.semilogx(permeability_array, N0_exact[:, 0, 0], label='full $\mathcal{N}^0[0,0]$', marker='+')
    plt.semilogx(permeability_array, N0_noproj[:, 0, 0], label='approx $\mathcal{N}^0[0,0]$', marker='^')
    plt.legend()

    # Plotting N0
    plt.figure()
    plt.semilogx(permeability_array, N0[:, 0, 0], label='approx $\mathcal{N}^0[0,0]$', marker='x')
    plt.semilogx(permeability_array, N0_exact[:, 0, 0], label='full $\mathcal{N}^0[0,0]$', marker='+')
    plt.semilogx(permeability_array, N0[:, 1, 1], label='approx $\mathcal{N}^0[1,1]$', marker='x')
    plt.semilogx(permeability_array, N0_exact[:, 1, 1], label='full $\mathcal{N}^0[1,1]$', marker='+')
    plt.semilogx(permeability_array, N0[:, 2, 2], label='approx $\mathcal{N}^0[2,2]$', marker='x')
    plt.semilogx(permeability_array, N0_exact[:, 2, 2], label='full $\mathcal{N}^0[2,2]$', marker='+')
    plt.semilogx(permeability_array, N0[:, 0, 1], label='approx $\mathcal{N}^0[0,1]$', marker='x')
    plt.semilogx(permeability_array, N0_exact[:, 0, 1], label='full $\mathcal{N}^0[0,1]$', marker='+')
    plt.semilogx(permeability_array, N0[:, 0, 2], label='approx $\mathcal{N}^0[0,2]$', marker='x')
    plt.semilogx(permeability_array, N0_exact[:, 0, 2], label='full $\mathcal{N}^0[0,2]$', marker='+')
    plt.semilogx(permeability_array, N0[:, 1, 2], label='approx $\mathcal{N}^0[1,2]$', marker='x')
    plt.semilogx(permeability_array, N0_exact[:, 1, 2], label='full $\mathcal{N}^0[1,2]$', marker='+')
    plt.xlabel('$\mu_r$')
    plt.ylabel('$\mathcal{N}^0$')
    plt.legend()

    # Plotting N0 Error
    plt.figure()
    diff = N0 - N0_exact
    diff_noproj = N0_noproj - N0_exact
    norm = np.asarray([np.linalg.norm(diff[ind, :, :], ord='fro') for ind in range(len(permeability_array))])
    norm_noproj = np.asarray([np.linalg.norm(diff_noproj[ind, :, :], ord='fro') for ind in range(len(permeability_array))])
    norm_exact = np.asarray([np.linalg.norm(N0[ind, :, :], ord='fro') for ind in range(len(permeability_array))])

    plt.loglog(permeability_array, norm / norm_exact, marker='x', label='Postprocessed')
    plt.loglog(permeability_array, norm_noproj / norm_exact, marker='+', label='Original')
    plt.legend()
    plt.ylabel('Fractional Error $\left(\mathcal{N}^0_{full}, \mathcal{N}^0_{NN} \\right)$')
    plt.xlabel('Relative Permeability')

    # Plotting Theta0 Solution Errors
    plt.figure()
    NRMSE = np.zeros((len(permeability_array), 3))
    for dim in range(3):
        for ind, mur in enumerate(permeability_array):
            numerator = (np.sum((D_pred[:, ind, dim] - D[:, ind, dim]) ** 2)) ** 0.5
            denominator = (np.sum(D[:, ind, dim] ** 2)) ** 0.5
            NRMSE[ind, dim] = numerator / denominator

    plt.loglog(permeability_array, NRMSE[:, 0], marker='x', label='NRMSE: dim=1', color='b')
    plt.loglog(permeability_array, NRMSE[:, 1], marker='+', label='NRMSE: dim=2', color='g')
    plt.loglog(permeability_array, NRMSE[:, 2], marker='^', label='NRMSE: dim=3', color='r')

    plt.legend()
    plt.ylabel('Fractional Error $\left(D_{full}, D_{NN} \\right)$')
    plt.xlabel('Relative Permeability')

    # Plotting Theta0 Truncation Errors
    plt.figure()
    NRMSE = np.zeros((len(permeability_array), 3))
    for dim in range(3):
        for ind, mur in enumerate(permeability_array):
            numerator = (np.sum((D_trunc[:, ind, dim] - D[:, ind, dim]) ** 2)) ** 0.5
            denominator = (np.sum(D[:, ind, dim] ** 2)) ** 0.5
            NRMSE[ind, dim] = numerator / denominator

    plt.loglog(permeability_array, NRMSE[:, 0], marker='x', label='NRMSE: dim=1')
    plt.loglog(permeability_array, NRMSE[:, 1], marker='+', label='NRMSE: dim=2')
    plt.loglog(permeability_array, NRMSE[:, 2], marker='^', label='NRMSE: dim=3')

    plt.legend()
    plt.ylabel('Fractional Error $\left(D_{full}, D_{trunc} \\right)$')
    plt.xlabel('Relative Permeability')

    # Plotting Theta0 NN Errors wrt truncated D
    plt.figure()
    NRMSE = np.zeros((len(permeability_array), 3))
    for dim in range(3):
        for ind, mur in enumerate(permeability_array):
            numerator = (np.sum((D_pred[:, ind, dim] - D_trunc[:, ind, dim]) ** 2)) ** 0.5
            denominator = (np.sum(D_trunc[:, ind, dim] ** 2)) ** 0.5
            NRMSE[ind, dim] = numerator / denominator

    plt.loglog(permeability_array, NRMSE[:, 0], marker='x', label='NRMSE: dim=1')
    plt.loglog(permeability_array, NRMSE[:, 1], marker='+', label='NRMSE: dim=2')
    plt.loglog(permeability_array, NRMSE[:, 2], marker='^', label='NRMSE: dim=3')

    plt.legend()
    plt.ylabel('Fractional Error $\left(D_{trunc}, D_{NN} \\right)$')
    plt.xlabel('Relative Permeability')

    # save_filename = 'Theta0Sol_sphere_0.25_log10_mur=0_to_2_21_samples_al=0.01_sig=1e6_220115' + '_trunc=1e-4'
    # save_all_figures('OutputFigures_220118', format='pickle', prefix=save_filename)
    # save_all_figures('OutputFigures_220118', format='png', prefix=save_filename)
    # save_all_figures('OutputFigures_220118', format='tex', prefix=save_filename)
    return D_pred, N0


def main_theta1():

    # loading fes and data from disk.
    theta0_solutions = np.load(r'/home/james/Desktop/MPT_calculator_James/Theta0Sol_Sphere_0.2_log10_mur=0_to_2_freq=1_to_8_25x26_samples_al=0.01_sig=1e6_220209.npy')
    theta1_solutions = np.load(r'/home/james/Desktop/MPT_calculator_James/Theta1Sol_Sphere_0.2_log10_mur=0_to_2_freq=1_to_8_25x26_samples_al=0.01_sig=1e6_220209.npy')
    fes1 = _load_fes_from_disk(r'/home/james/Desktop/MPT_calculator_James/FES_Sphere_0.2_log10_mur=0_to_2_freq=1_to_8_25x26_samples_al=0.01_sig=1e6_220209.pkl')
    fes2 = _load_fes_from_disk(r'/home/james/Desktop/MPT_calculator_James/FES2_Sphere_0.2_log10_mur=0_to_2_freq=1_to_8_25x26_samples_al=0.01_sig=1e6_220209.pkl')

    permeability_array = np.logspace(0,2,25)
    frequency_array = np.logspace(1, 8, 26)
    permeability_query = permeability_array
    frequency_query = frequency_array

    # permeability_array = np.asarray([permeability_array[24]])
    # frequency_array = np.asarray([frequency_array[15]])

    n_perm = len(permeability_array)
    n_freq = len(frequency_array)

    # reshaping theta1 solutions into ndofx3xn^2 array.
    theta1_solutions = np.reshape(theta1_solutions, (fes2.ndof, 3, n_perm*n_freq))
    theta1_pred = np.zeros(theta1_solutions.shape, dtype=complex)
    #
    # # Applying postprocessing.
    # proj = _define_postprojection(fes2)
    # for dim in range(3):
    #     theta1 = GridFunction(fes2)
    #     for ind in range(15*16):
    #         theta1.vec.FV().NumPy()[:] = theta1_solutions[:, dim, ind]
    #         theta1.vec.data = proj * (theta1.vec)
    #         theta1_solutions[:,dim,  ind] = theta1.vec.FV().NumPy()[:]

    theta0_pred, N0 = main_theta0()
    plt.close('all')
    N0 = np.vstack((np.zeros((1,3,3)), N0))

    plt.figure()
    for dim in range(3):
        U_trunc, S_trunc, Vh_trunc = truncated_SVD(theta1_solutions[:,dim,:], tol=1e-4)
        plt.figure()
        plt.semilogy(np.diag(S_trunc)/S_trunc[0,0], marker='x', label=f'dir = {dim}')
        plt.xlabel('Mode')
        plt.ylabel('Normalised Singular Values')

        real_Vh = Vh_trunc.real
        imag_Vh = Vh_trunc.imag

        real_vh_pred = np.zeros((Vh_trunc.shape[0], len(frequency_query)*len(permeability_query)))
        imag_vh_pred = np.zeros((Vh_trunc.shape[0], len(frequency_query)*len(permeability_query)))

        for ind in range(Vh_trunc.shape[0]):  # looping through each retained modes.
            real_mode = real_Vh[ind,:]
            imag_mode = imag_Vh[ind,:]

            real_vh_pred[ind,:] = neural_network_2D(np.log10(permeability_array.reshape((n_perm, 1))),
                                                    np.log10(frequency_array.reshape((n_freq,1))),
                                                    real_mode.reshape((n_perm*n_freq,1)),
                                                    np.log10(permeability_query.reshape((n_perm,1))),
                                                    np.log10(frequency_query.reshape((n_freq,1))),
                                                    scaler='standard',
                                                    activation='tanh')

            imag_vh_pred[ind, :] = neural_network_2D(np.log10(permeability_array.reshape((n_perm, 1))),
                                                    np.log10(frequency_array.reshape((n_freq,1))),
                                                    imag_mode.reshape((n_perm*n_freq,1)),
                                                    np.log10(permeability_query.reshape((n_perm,1))),
                                                    np.log10(frequency_query.reshape((n_freq,1))),
                                                    scaler='standard',
                                                    activation='tanh')

            print(f'solved for mode {ind+1} / {Vh_trunc.shape[0]} in dimension {dim+1}')

        Vh_pred = real_vh_pred + 1j*imag_vh_pred
        theta1_pred[:,dim,:] = U_trunc @ S_trunc @ Vh_pred
        plt.close('all')

    theta1_approx = theta1_pred.reshape((fes2.ndof, 3, n_perm, n_freq))

    # Calculating MPT coefficients.
    mesh = fes1.mesh
    R = np.empty((len(permeability_array), len(frequency_array), 3, 3))
    I = np.empty((len(permeability_array), len(frequency_array), 3, 3))
    MPT = np.empty((len(permeability_array), len(frequency_array), 3, 3), dtype=complex)
    xivec = [CoefficientFunction((0, -z, y)), CoefficientFunction((z, 0, -x)), CoefficientFunction((-y, x, 0))]
    alpha = 0.01
    inorout = {'air': 0, 'sphere': 1}
    inout_coef = [inorout[mat] for mat in mesh.GetMaterials()]
    inout = CoefficientFunction(inout_coef)
    sig = {'air': 0.0, 'sphere': 1e6}
    sig_coeff = CoefficientFunction([sig[mat] for mat in fes2.mesh.GetMaterials()])

    # N0, _ = calc_N0_from_D(theta0_solutions, fes1, alpha, permeability_array)
    # N0 = np.asarray(N0)


    for ind, mu in enumerate(permeability_array):
        mur = {'air': 1.0, 'sphere': mu}
        mur_coeff = CoefficientFunction([mur[mat] for mat in fes2.mesh.GetMaterials()])
        for jnd, omega in enumerate(frequency_array):
            nu = omega * 4 * np.pi * 1e-7 * (alpha ** 2)
            c1, c2 = calc_MPT_coeffs(fes1, fes2, theta1_approx[:,:,ind, jnd], theta0_solutions[:,ind,:],xivec,alpha,mur_coeff,sig_coeff,inout,nu )
            R[ind,jnd, :, :] = c1
            I[ind,jnd, :, :] = c2

            MPT[ind,jnd, :, :] = N0[ind,:,:] + R[ind,jnd,:,:] + 1j*I[ind,jnd,:,:]
            print(f'solved MPT for perm {ind} and freq {jnd}')



    return MPT


def test_2d_NN():
    x = np.linspace(0,2*np.pi,20)
    y = np.linspace(-np.pi,np.pi,20)
    z = np.sin(x[:,None]) @ np.sin(y[:,None]).transpose()

    plt.figure(); plt.imshow(z.reshape((len(x),len(y))))

    z_q = neural_network_2D(x.reshape((len(x), 1)),
                      y.reshape((len(y), 1)),
                      z.reshape((len(x) * len(y), 1)),
                      x.reshape((len(x), 1)),
                      y.reshape((len(y), 1)),
                            scaler='minmax',
                            activation='tanh')

    return z_q


def eval_truncation_accuracy():
    filenames = [
        r'C:\Users\James\Desktop\MPT_calculator_James\Theta0Sol_sphere_0.08_log10_mur=0_to_2_11_samples_al=0.01_sig=1e6_220116.npy',
        r'C:\Users\James\Desktop\MPT_calculator_James\Theta0Sol_sphere_0.08_log10_mur=0_to_2_21_samples_al=0.01_sig=1e6_220116.npy',
        r'C:\Users\James\Desktop\MPT_calculator_James\Theta0Sol_sphere_0.08_log10_mur=0_to_2_31_samples_al=0.01_sig=1e6_220116.npy',
        r'C:\Users\James\Desktop\MPT_calculator_James\Theta0Sol_sphere_0.08_log10_mur=0_to_2_41_samples_al=0.01_sig=1e6_220116.npy',
        r'C:\Users\James\Desktop\MPT_calculator_James\Theta0Sol_sphere_0.08_log10_mur=0_to_2_51_samples_al=0.01_sig=1e6_220116.npy']

    for filename in filenames:
        D_full = _load_D_from_disk(filename)
        D_full = D_full[:, 1:, :]
        N_snapshots = D_full.shape[1]
        D_reduced = np.zeros(D_full.shape)
        frac_norm = []

        plt.figure(1)
        plt.figure(2)

        for ind, N_modes in enumerate([1, 2, 4, 8, 16, 32, 50]):
            for dim in [0, 1, 2]:
                D = D_full[:, :, dim]
                U_trunc, S_trunc, Vh_trunc = truncated_SVD(D, 1e-3, n_elements=N_modes)
                D_reduced[:, :, dim] = U_trunc @ S_trunc @ Vh_trunc

            diff = D_full - D_reduced
            frac_norm += [np.linalg.norm(diff) / np.linalg.norm(D_full)]
        plt.loglog([1, 2, 4, 8, 16, 32, 50], frac_norm, marker='x', basex=2, label=f'N snapshots = {N_snapshots}')
        plt.xlabel('N Modes')
        plt.ylabel('$\\frac{||D - D_{reduced}||_F}{||D||_F}$')
        plt.legend()


def ben_theta_solutions():
    permeability_array = np.logspace(1, 1, 1)
    frequency_array = np.logspace(3,3,1)

    # Solver = "bddc"
    # epsi = 10 ** -12
    # Maxsteps = 2500
    # Tolerance = 10 ** -8

    Object = 'sphere.geo'
    alpha = 0.01
    Order = 3
    inorout = {'air': 0, 'sphere': 1}
    sig = {'air': 0.0, 'sphere': 1e6}

    for ind, perm in enumerate(permeability_array):
        for jnd, freq in enumerate(frequency_array):
            mur = {'air': 1, 'sphere': perm}

            # Calling Ben's version of single solve
            Theta0Sol, Theta1Sol = SingleSolve.SingleFrequency(Object, Order, alpha, inorout, mur, sig, freq, 1, False,
                                                               False, curve=5, theta_solutions_only=False)

            if ind == 0 and jnd == 0:
                D0 = np.zeros((len(Theta0Sol[:, 0]), 3, len(permeability_array), len(frequency_array)))
                D1 = np.zeros((len(Theta1Sol[:, 0]), 3, len(permeability_array), len(frequency_array)), dtype=complex)

            D0[:, :, ind, jnd] = Theta0Sol
            D1[:, :, ind, jnd] = Theta1Sol

    return D0, D1


def quick_calc_MPT():
    theta0_solutions = np.load(
        r'C:\Users\James\Desktop\MPT_calculator_James\Theta0Sol_Test.npy')
    theta1_solutions = np.load(
        r'C:\Users\James\Desktop\MPT_calculator_James\Theta1Sol_Test.npy')
    fes1 = _load_fes_from_disk(
        r'C:\Users\James\Desktop\MPT_calculator_James\FES_Test.pkl')
    fes2 = _load_fes_from_disk(
        r'C:\Users\James\Desktop\MPT_calculator_James\FES2_Test.pkl')

    mesh = fes1.mesh
    Solver, epsi, Maxsteps, Tolerance = Settings.SolverParameters()
    alpha = 0.01
    Order = 3
    inorout = {'air': 0, 'sphere': 1}
    inout_coef = [inorout[mat] for mat in mesh.GetMaterials()]
    inout = CoefficientFunction(inout_coef)
    sig = {'air': 0.0, 'sphere': 1e6}
    # fes = HCurl(mesh, order=Order, dirichlet="outer", flags={"nograds": True})
    evec = [ CoefficientFunction( (1,0,0) ), CoefficientFunction( (0,1,0) ), CoefficientFunction( (0,0,1) ) ]
    mur = {'air': 1.0, 'sphere': 1.5}
    mur_coeff = CoefficientFunction([mur[mat] for mat in fes2.mesh.GetMaterials()])

    u, v = fes2.TnT()
    w, z = fes1.TnT()

    K2 = BilinearForm(fes2, symmetric=True)
    K2 += SymbolicBFI(inout * curl(u) * curl(v))
    K2.Assemble()

    K1 = BilinearForm(fes2, symmetric=True)
    K1 += SymbolicBFI((1-inout) * curl(u) * curl(v))
    K1.Assemble()

    # Ut = GridFunction(fes2)
    # U = GridFunction(fes2)
    # Ut.vec.FV().NumPy()[:] = np.squeeze(theta1_solutions[:,0,:,0])[None,:]
    # U.vec.FV().NumPy()[:] = np.squeeze(theta1_solutions[:,0,:,0])[:,None]

    rows,cols,vals = (K1.mat).COO()
    A1 = sp.csr_matrix((vals,(rows,cols)))
    rows, cols, vals = (K2.mat).COO()
    A2 = sp.csr_matrix((vals, (rows, cols)))
    A = A1 + A2/1.5
    R = np.squeeze(theta1_solutions[:,0,:,0])[None,:] @ A @ np.conj(np.squeeze(theta1_solutions[:,0,:,0])[:,None])
    R *= -(alpha**3)/4
    print(R)




if __name__ == '__main__':
    # fes = _load_fes_from_disk(
    # r'C:\Users\James\Desktop\MPT_calculator_James\FES_sphere_0.08_log10_mur=0_to_2_21_samples_al=0.01_sig=1e6_220116.pkl')
    # main()
    # _svd_comparison()
    quick_calc_MPT()

    # main_theta0()
    # _plot_theta0_sols(plot_curl=False, proj_mask=False)
    # _plot_theta0_sols(proj_mask=True, fig=plt.gcf(), plot_curl=True)
    # eval_truncation_accuracy()
    # performance_matrix = _eval_nn_hyperparameters()

    # os.chdir('..')

    # os.chdir('..')
    # theta0sol, theta1sol = ben_theta_solutions()
    #
    # np.save('Theta0_sphere_0.2_BenTest_freq=1-8-7_mur=0-2-6', theta0sol)
    # np.save('Theta1_sphere_0.2_BenTest_freq=1-8-7_mur=0-2-6', theta1sol)
    #
    # D1_full = np.load(r'C:\Users\James\Desktop\MPT_calculator_James\Theta1Sol_Sphere_0.2_log10_mur=1_to_2_10x10_samples_al=0.01_sig=1e6_220128.npy')
    # D0_full = np.load(r'C:\Users\James\Desktop\MPT_calculator_James\Theta0Sol_Sphere_0.2_log10_mur=1_to_2_10x10_samples_al=0.01_sig=1e6_220128.npy')
    # fes = _load_fes_from_disk(r'C:\Users\James\Desktop\MPT_calculator_James\FES2_Sphere_0.2_log10_mur=1_to_2_10x10_samples_al=0.01_sig=1e6_220128.pkl')
    #
    # xx, yy = np.meshgrid(np.linspace(1,8,10), np.linspace(0,2,10))
    #
    # L2_norm = np.empty(xx.shape)
    # theta_1 = GridFunction(fes)
    #
    # for dim in range(3):
    #     for ind in range(10):
    #         for jnd in range(10):
    #             D = D1_full[:,dim,ind,jnd]
    #             theta_1.vec.FV().NumPy()[:] = D
    #             integrand = InnerProduct(theta_1.real, theta_1.real)
    #             L2_norm[ind, jnd] = Integrate(integrand, fes.mesh) ** 0.5
    #
    #     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #     ax.plot_surface(xx,yy, L2_norm)
    # #
    # # # comparison
    # # D1_Ben = np.load(r'C:\Users\James\Desktop\MPT_calculator_James\Theta1_sphere_0.2_BenTest_freq=1-8-7_mur=0-2-6.npy')
    # # D1_diff = D1_full - D1_Ben
    #
    # MPT = main_theta1()
    # # zq = test_2d_NN()
    # MPT11 = np.zeros((25,26), dtype=complex)
    # for ind in range(25):
    #     for jnd in range(26):
    #         MPT11[ind, jnd] = np.linalg.eigvals(MPT[ind,jnd,:,:])[0]
    #
    # permeability_array = np.linspace(1,100,25)
    # frequency_array = np.linspace(1, 8, 26)
    # xx, yy = np.meshgrid(permeability_array, frequency_array)
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot_surface(xx, yy, MPT11.real)
    # plt.xlabel('$\mu_r$')
    # plt.ylabel('$\mathrm{log}_{10}(\omega)$')
    # ax.set_zlabel('$\mathrm{re}(\lambda_1)$')
    #
    #
    # fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    # ax2.plot_surface(xx, yy, MPT11.imag)
    # plt.xlabel('$\mu_r$')
    # plt.ylabel('$\mathrm{log}_{10}(\omega)$')
    # ax2.set_zlabel('$\mathrm{im}(\lambda_1)$')
    # # ax2.set_zlim([0,5e-6])
    #



