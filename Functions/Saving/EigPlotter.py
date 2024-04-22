#Importing
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from .TickFormatter import *

#This is required for when a copy of the file is sent to the results section
try:
    sys.path.insert(0,"Settings")
except:
    pass

from PlotterSettings import PlotterSettings


def EigPlotter(savename, Array, EigenValues, EddyCurrentTest):
    """
    B.A. Wilson, P.D. Ledger, J. Elgy 2020-2022.
    Function to plot and save eigenvalues as function of frequency.

    Args:
        savename (str): path to save figures
        Array (list): frequency array
        EigenValues (np.ndarray): Nx3 list of complex eigenvalues
        EddyCurrentTest (float | None): if using eddy current test, max frequency, else None

    Returns:
        bool: plot figure or not.
    """
    
    
    # Create a way to reference xkcd colours
    PYCOL = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
             '#17becf']

    # Retrieve the settings for the plot
    Title, Show, ETP, _, MLS, MMS, _, _, _, _, ECL = PlotterSettings()

    # Plot the graph
    fig, ax = plt.subplots()

    # Plot the mainlines
    for i, line in enumerate(ETP):
        if i == 0:
            lines = ax.plot(Array, EigenValues[:, line - 1].real, MLS, markersize=MMS, color=PYCOL[i])
        else:
            lines += ax.plot(Array, EigenValues[:, line - 1].real, MLS, markersize=MMS, color=PYCOL[i])

    ymin, ymax = ax.get_ylim()

    # Format the axes
    plt.xscale('log')
    plt.ylim(ymin, ymax)
    ax.grid(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(TickFormatter))
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.94, top=0.90)

    # Label the axes
    plt.xlabel("Frequency (rad/s)")
    plt.ylabel(r"$\lambda(\mathcal{N}^0+\mathcal{R})$")

    # Title
    if Title == True:
        plt.title(r"Eigenvalues of $\mathcal{N}^0+\mathcal{R}$")

    # Create the legend
    names = []
    for i, number in enumerate(ETP):
        names.append(r"$\lambda_{" + str(number) + "}(\mathcal{N}^0+\mathcal{R})$")

    # Show where the eddy-current breaks down (if applicable)
    if isinstance(EddyCurrentTest, float):
        if ECL == True:
            x = np.ones(10) * EddyCurrentTest
            y = np.linspace(ymin, ymax, 10)
            lines += ax.plot(x, y, '--r')
            names.append(r"eddy-current model valid")

    # Make the legend
    ax.legend(lines, names)

    # Save the graph
    plt.savefig(savename + "RealEigenvalues.pdf")

    # Plot the imaginary graph
    fig, ax = plt.subplots()

    # Plot the mainlines
    for i, line in enumerate(ETP):
        if i == 0:
            lines = ax.plot(Array, EigenValues[:, line - 1].imag, MLS, markersize=MMS, color=PYCOL[i])
        else:
            lines += ax.plot(Array, EigenValues[:, line - 1].imag, MLS, markersize=MMS, color=PYCOL[i])

    ymin, ymax = ax.get_ylim()

    # Format the axes
    plt.xscale('log')
    plt.ylim(ymin, ymax)
    ax.grid(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(TickFormatter))
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.94, top=0.90)

    # Label the axes
    plt.xlabel("Frequency (rad/s)")
    plt.ylabel(r"$\lambda(\mathcal{I})$")

    # Title
    if Title == True:
        plt.title(r"Eigenvalues of $\mathcal{I}$")

    # Create the legend
    names = []
    for i, number in enumerate(ETP):
        names.append(r"$\lambda_{" + str(number) + "}(\mathcal{I})$")

    # Show where the eddy-current breaks down (if applicable)
    if isinstance(EddyCurrentTest, float):
        if ECL == True:
            x = np.ones(10) * EddyCurrentTest
            y = np.linspace(ymin, ymax, 10)
            lines += ax.plot(x, y, '--r')
            names.append(r"eddy-current model valid")

    # Make the legend
    ax.legend(lines, names)

    # Save the graph
    plt.savefig(savename + "ImaginaryEigenvalues.pdf")

    return Show
