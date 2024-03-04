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


def PODErrorPlotter(savename, Array, PODArray, Values, PODValues, Errors, EddyCurrentTest):
    # Create a way to reference xkcd colours
    PYCOL = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
             '#17becf']

    # Retrieve the settings for the plot
    Title, Show, _, TTP, MLS, MMS, SLS, SMS, EBLS, EBMS, ECL = PlotterSettings()

    # Plot the graph
    fig, ax = plt.subplots()

    # Plot the mainlines
    for i, line in enumerate(TTP):
        if i == 0:
            # Plot main lines
            lines = ax.plot(Array, Values[:, line - 1].real, MLS, markersize=MMS, color=PYCOL[i])
            # Plot bottom error bars
            lines += ax.plot(Array, Values[:, line - 1].real - Errors[:, line - 1], EBLS, markersize=EBMS,
                             color=PYCOL[i])
        else:
            # Plot main lines
            lines += ax.plot(Array, Values[:, line - 1].real, MLS, markersize=MMS, color=PYCOL[i])
            # Plot bottom error bars
            lines += ax.plot(Array, Values[:, line - 1].real - Errors[:, line - 1], EBLS, markersize=EBMS,
                             color=PYCOL[i])

    # Plot the snapshots
    for i, line in enumerate(TTP):
        lines += ax.plot(PODArray, PODValues[:, line - 1].real, SLS, markersize=SMS, color=PYCOL[i])

    # Calculate the limits
    ymin = min(np.amin(Values.real), np.amin(PODValues.real))
    ymax = max(np.amax(Values.real), np.amax(PODValues.real))
    y_range = ymax - ymin
    ymin -= 0.05 * y_range
    ymax += 0.05 * y_range

    # Show where the eddy-current breaks down (if applicable)
    if isinstance(EddyCurrentTest, float):
        if ECL == True:
            x = np.ones(10) * EddyCurrentTest
            y = np.linspace(ymin, ymax, 10)
            lines += ax.plot(x, y, '--r')

    # Plot the top error bars (we plot them seperatly for the legend order)
    for i, line in enumerate(TTP):
        lines += ax.plot(Array, Values[:, line - 1].real + Errors[:, line - 1], EBLS, markersize=EBMS, color=PYCOL[i])

    # Format the axes
    plt.xscale('log')
    plt.ylim(ymin, ymax)
    ax.grid(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(TickFormatter))
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.94, top=0.90)

    # Label the axes
    plt.xlabel("Frequency (rad/s)")
    plt.ylabel(r"$\mathcal{N}^0_{ij}+\mathcal{R}_{ij}$")

    if Title == True:
        plt.title(r"Tensor coefficients of $\mathcal{N}^0+\mathcal{R}$")

    # Create the legend
    names = []
    CoefficientRef = ["11", "12", "13", "22", "23", "33", "21", "31", "_", "32"]
    for i, number in enumerate(TTP):
        if number == 1 or number == 4 or number == 6:
            names.append(r"Re($\mathcal{M}_{" + CoefficientRef[number - 1] + "}(\omega)$)")
            names.append(r"Re($\mathcal{M}_{" + CoefficientRef[number - 1] + "}(\omega)$) (Certificate Bound)")
        else:
            names.append(
                r"Re($\mathcal{M}_{" + CoefficientRef[number - 1] + "}(\omega)$)=Re($\mathcal{M}_{" + CoefficientRef[
                    number + 4] + "}(\omega)$)")
            names.append(
                r"Re($\mathcal{M}_{" + CoefficientRef[number - 1] + "}(\omega)$)=Re($\mathcal{M}_{" + CoefficientRef[
                    number + 4] + "}(\omega)$) (Certificate Bound)")

    for i, number in enumerate(TTP):
        if number == 1 or number == 4 or number == 6:
            names.append(r"Re($\mathcal{M}_{" + CoefficientRef[number - 1] + "}(\omega)$) (Snapshot)")
        else:
            names.append(
                r"Re($\mathcal{M}_{" + CoefficientRef[number - 1] + "}(\omega)$)=Re($\mathcal{M}_{" + CoefficientRef[
                    number + 4] + "}(\omega)$) (Snapshot)")

    if isinstance(EddyCurrentTest, float):
        if ECL == True:
            names.append(r"eddy-current model valid")

    # Shrink the size of the legend if there are to many lines
    if len(names) > 13:
        ax.legend(lines, names, prop={'size': 5})
    elif len(names) > 7:
        ax.legend(lines, names, prop={'size': 8})
    else:
        ax.legend(lines, names)

    # Save the graph
    plt.savefig(savename + "RealTensorCoeficients.pdf")

    # Plot the imaginary graph
    fig, ax = plt.subplots()

    # Plot the mainlines
    for i, line in enumerate(TTP):
        if i == 0:
            # Plot the main lines
            lines = ax.plot(Array, Values[:, line - 1].imag, MLS, markersize=MMS, color=PYCOL[i])
            # Plot bottom error bars
            lines += ax.plot(Array, Values[:, line - 1].imag - Errors[:, line - 1], EBLS, markersize=EBMS,
                             color=PYCOL[i])
        else:
            # Plot the main lines
            lines += ax.plot(Array, Values[:, line - 1].imag, MLS, markersize=MMS, color=PYCOL[i])
            # Plot bottom error bars
            lines += ax.plot(Array, Values[:, line - 1].imag - Errors[:, line - 1], EBLS, markersize=EBMS,
                             color=PYCOL[i])

    # Plot the snapshots
    for i, line in enumerate(TTP):
        lines += ax.plot(PODArray, PODValues[:, line - 1].imag, SLS, markersize=SMS, color=PYCOL[i])

    # Calculate the limits
    ymin = min(np.amin(Values.imag), np.amin(PODValues.imag))
    ymax = max(np.amax(Values.imag), np.amax(PODValues.imag))
    y_range = ymax - ymin
    ymin -= 0.05 * y_range
    ymax += 0.05 * y_range

    # Show where the eddy-current breaks down (if applicable)
    if isinstance(EddyCurrentTest, float):
        if ECL == True:
            x = np.ones(10) * EddyCurrentTest
            y = np.linspace(ymin, ymax, 10)
            lines += ax.plot(x, y, '--r')

    # Plot the top error bars (we plot them seperatly for the legend order)
    for i, line in enumerate(TTP):
        lines += ax.plot(Array, Values[:, line - 1].imag + Errors[:, line - 1], EBLS, markersize=EBMS, color=PYCOL[i])

    # Format the axes
    plt.xscale('log')
    plt.ylim(ymin, ymax)
    ax.grid(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(TickFormatter))
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.94, top=0.90)

    # Label the axes
    plt.xlabel("Frequency (rad/s)")
    plt.ylabel(r"$\mathcal{I}_{ij}$")

    if Title == True:
        plt.title(r"Tensor coefficients of $\mathcal{I}$")

    # Create the legend
    names = []
    for i, number in enumerate(TTP):
        if number == 1 or number == 4 or number == 6:
            names.append(r"Im($\mathcal{M}_{" + CoefficientRef[number - 1] + "}(\omega)$)")
            names.append(r"Im($\mathcal{M}_{" + CoefficientRef[number - 1] + "}(\omega)$) (Certificate Bound)")
        else:
            names.append(
                r"Im($\mathcal{M}_{" + CoefficientRef[number - 1] + "}(\omega)$)=Im($\mathcal{M}_{" + CoefficientRef[
                    number + 4] + "}(\omega)$)")
            names.append(
                r"Im($\mathcal{M}_{" + CoefficientRef[number - 1] + "}(\omega)$)=Im($\mathcal{M}_{" + CoefficientRef[
                    number + 4] + "}(\omega)$) (Certificate Bound)")
    for i, number in enumerate(TTP):
        if number == 1 or number == 4 or number == 6:
            names.append(r"Im($\mathcal{M}_{" + CoefficientRef[number - 1] + "}(\omega)$) (Snapshot)")
        else:
            names.append(
                r"Im($\mathcal{M}_{" + CoefficientRef[number - 1] + "}(\omega)$)=Im($\mathcal{M}_{" + CoefficientRef[
                    number + 4] + "}(\omega)$) (Snapshot)")

    # Show where the eddy-current breaks down (if applicable)
    if isinstance(EddyCurrentTest, float):
        if ECL == True:
            names.append(r"eddy-current model valid")

    # Shrink the size of the legend if there are to many lines
    if len(names) > 13:
        ax.legend(lines, names, prop={'size': 5})
    elif len(names) > 7:
        ax.legend(lines, names, prop={'size': 8})
    else:
        ax.legend(lines, names)

    # Save the graph
    plt.savefig(savename + "ImaginaryTensorCoeficients.pdf")

    return Show



