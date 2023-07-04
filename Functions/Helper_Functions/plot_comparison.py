# James Elgy - 04/07/2023

import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter.filedialog import askdirectory as ask_dir

def compare_two_datasets(comparison='tensors', colors='default', legend='default'):
    """
    James Elgy 2023:
    Small tool for plotting comparisons between MPT sweeps. This isn't particularly sophisticated, and for more indepth
    comparisons, bespoke plotting scripts should be written by the user.

    Function works by loading either Tensors.csv or Eigenvalues.csv from two chosen sweep results directories

    Parameters
    ----------
    comparison: str. Either 'tensors' or 'eigenvalues'. Controls which is plotted.
    colors: list of strings containing matplotlib colors. e.g. ['b', 'r']. 'default' uses ['b', 'r']
    legend: list of strings containing legend entries for each dataset. 'default' uses ['sweep 1', 'sweep 2']

    """
    comparison = comparison.lower()
    if colors == 'default':
        colors = ['b', 'r']
    if legend == 'default':
        legend = ['sweep 1', 'sweep 2']

    # Loading datasets for comparison
    root = tk.Tk()
    root.withdraw()
    folder1 = ask_dir(title='Select dataset 1')
    folder2 = ask_dir(title='Select dataset 2')

    if comparison == 'tensors':
        data1 = np.genfromtxt(folder1 + '/Data/Tensors.csv', dtype=complex, delimiter=', ')
        data2 = np.genfromtxt(folder2 + '/Data/Tensors.csv', dtype=complex, delimiter=', ')
        PODdata1 = np.genfromtxt(folder1 + '/Data/PODTensors.csv', dtype=complex, delimiter=', ')
        PODdata2 = np.genfromtxt(folder2 + '/Data/PODTensors.csv', dtype=complex, delimiter=', ')
    elif comparison == 'eigenvalues':
        data1 = np.genfromtxt(folder1 + '/Data/Eigenvalues.csv', delimiter=', ', dtype=complex)
        data2 = np.genfromtxt(folder2 + '/Data/Eigenvalues.csv', delimiter=', ', dtype=complex)
        PODdata1 = np.genfromtxt(folder1 + '/Data/PODEigenvalues.csv', delimiter=', ', dtype=complex)
        PODdata2 = np.genfromtxt(folder2 + '/Data/PODEigenvalues.csv', delimiter=', ', dtype=complex)

    freqs1 = np.genfromtxt(folder1 + '/Data/Frequencies.csv')
    PODfreqs1 = np.genfromtxt(folder1 + '/Data/PODFrequencies.csv')
    freqs2 = np.genfromtxt(folder2 + '/Data/Frequencies.csv')
    PODfreqs2 = np.genfromtxt(folder2 + '/Data/PODFrequencies.csv')

    # Plotting compariosn:

    plt.figure()
    for d, (data, PODdata, freqs, PODfreqs) in enumerate(zip([data1, data2], [PODdata1, PODdata2], [freqs1, freqs2], [PODfreqs1, PODfreqs2])):
        for i in range(data.shape[1]):
            if i == 0:
                plt.semilogx(freqs, data[:,i].real, color=colors[d], label=legend[d])
            else:
                plt.semilogx(freqs, data[:, i].real, color=colors[d])
            plt.semilogx(PODfreqs, PODdata[:,i].real, color=colors[d], linestyle='None', marker='x')
    plt.legend()
    plt.xlabel('$\omega$ [rad/s]')
    if comparison == 'tensors':
        plt.ylabel(r'$(\tilde{\mathcal{R}})_{ij}$ [m$^3$]')
    elif comparison == 'eigenvalues':
        plt.ylabel(r'$\lambda_i(\tilde{\mathcal{R}})$ [m$^3$]')

    plt.figure()
    for d, (data, PODdata) in enumerate(zip([data1, data2], [PODdata1, PODdata2])):
        for i in range(data.shape[1]):
            if i == 0:
                plt.semilogx(freqs, data[:,i].imag, color=colors[d], label=legend[d])
            else:
                plt.semilogx(freqs, data[:, i].imag, color=colors[d])
            plt.semilogx(PODfreqs, PODdata[:,i].imag, color=colors[d], linestyle='None', marker='x')
    plt.legend()
    plt.xlabel('$\omega$ [rad/s]')
    if comparison == 'tensors':
        plt.ylabel(r'$(\mathcal{I})_{ij}$ [m$^3$]')
    elif comparison == 'eigenvalues':
        plt.ylabel(r'$\lambda_i({\mathcal{I}})$ [m$^3$]')

    plt.show()

if __name__ == '__main__':
    compare_two_datasets(comparison='eigenvalues')
