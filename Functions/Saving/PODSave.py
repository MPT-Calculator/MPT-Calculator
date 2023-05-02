import os
import sys
from math import floor, log10
import numpy as np
from shutil import copyfile
from zipfile import *

import netgen.meshing as ngmeshing
from ngsolve import Mesh

sys.path.insert(0,"Functions")
from Settings import SaverSettings
from .PODEigPlotter import *
from .PODTensorPlotter import *
from .PODErrorPlotter import *
from .FtoS import *
from .DictionaryList import *


def PODSave(Geometry, Array, TensorArray, EigenValues, N0, PODTensors, PODEigenValues, PODArray, PODTol, elements,
            alpha, Order, MeshSize, mur, sig, ErrorTensors, EddyCurrentTest, invariants, using_iterative_POD=False):
    # Find how the user wants the data to be saved
    # FolderStructure = SaverSettings()

    # Create a temp folder in the results directory.
    FolderStructure = 'Default'

    if FolderStructure == "Default":
        # Create the file structure
        # Define constants for the folder name
        objname = Geometry[:-4]
        minF = Array[0]
        strminF = FtoS(minF)
        maxF = Array[-1]
        strmaxF = FtoS(maxF)
        Points = len(Array)
        PODPoints = len(PODArray)
        strmur = DictionaryList(mur, False)
        strsig = DictionaryList(sig, True)
        strPODTol = FtoS(PODTol)

        # Define the main folder structure
        subfolder1 = "al_" + str(alpha) + "_mu_" + strmur + "_sig_" + strsig
        subfolder2 = strminF + "-" + strmaxF + "_" + str(Points) + "_el_" + str(elements) + "_ord_" + str(
            Order) + "_POD_" + str(PODPoints) + "_" + strPODTol
        if using_iterative_POD is True:
            subfolder2 += '_Iterative_POD'
        sweepname = objname + "/" + subfolder1 + "/" + subfolder2
    else:
        sweepname = FolderStructure

    # Save the data
    np.savetxt("Results/" + sweepname + "/Data/Frequencies.csv", Array, delimiter=",")
    np.savetxt("Results/" + sweepname + "/Data/PODFrequencies.csv", PODArray, delimiter=",")
    np.savetxt("Results/" + sweepname + "/Data/Eigenvalues.csv", EigenValues, delimiter=",")
    np.savetxt("Results/" + sweepname + "/Data/PODEigenvalues.csv", PODEigenValues, delimiter=",")
    np.savetxt("Results/" + sweepname + "/Data/N0.csv", N0, delimiter=",")
    np.savetxt("Results/" + sweepname + "/Data/Tensors.csv", TensorArray, delimiter=",")
    np.savetxt("Results/" + sweepname + "/Data/PODTensors.csv", PODTensors, delimiter=",")
    np.savetxt("Results/" + sweepname + "/Data/Invariants.csv", invariants, delimiter=",")

    if isinstance(EddyCurrentTest, float):
        f = open('Results/' + sweepname + '/Data/Eddy-current_breakdown.txt', 'w+')
        f.write('omega = ' + str(round(EddyCurrentTest)))
        f.close()

    # Format the tensor arrays so they can be plotted
    PlottingTensorArray = np.zeros([Points, 6], dtype=complex)
    PlottingPODTensors = np.zeros([PODPoints, 6], dtype=complex)
    PlottingTensorArray = np.concatenate(
        [np.concatenate([TensorArray[:, :3], TensorArray[:, 4:6]], axis=1), TensorArray[:, 8:9]], axis=1)
    PlottingPODTensors = np.concatenate(
        [np.concatenate([PODTensors[:, :3], PODTensors[:, 4:6]], axis=1), PODTensors[:, 8:9]], axis=1)
    try:
        ErrorTensors[:, [1, 3]] = ErrorTensors[:, [3, 1]]
        ErrorTensors[:, [2, 4]] = ErrorTensors[:, [4, 2]]
        ErrorTensors[:, [4, 5]] = ErrorTensors[:, [5, 4]]
    except:
        pass

    # Define where to save the graphs
    savename = "Results/" + sweepname + "/Graphs/"

    # Plot the graphs
    Show = PODEigPlotter(savename, Array, PODArray, EigenValues, PODEigenValues, EddyCurrentTest)

    try:
        if ErrorTensors == False:
            Show = PODTensorPlotter(savename, Array, PODArray, PlottingTensorArray, PlottingPODTensors, EddyCurrentTest)
    except:
        Show = PODErrorPlotter(savename, Array, PODArray, PlottingTensorArray, PlottingPODTensors, ErrorTensors,
                               EddyCurrentTest)

        # Change the format of the error bars to the format of the Tensors
        Errors = np.zeros([Points, 9])
        Errors[:, 0] = ErrorTensors[:, 0]
        Errors[:, 1] = ErrorTensors[:, 3]
        Errors[:, 2] = ErrorTensors[:, 4]
        Errors[:, 3] = ErrorTensors[:, 3]
        Errors[:, 4] = ErrorTensors[:, 1]
        Errors[:, 5] = ErrorTensors[:, 5]
        Errors[:, 6] = ErrorTensors[:, 4]
        Errors[:, 7] = ErrorTensors[:, 5]
        Errors[:, 8] = ErrorTensors[:, 2]
        np.savetxt("Results/" + sweepname + "/Data/ErrorBars.csv", Errors, delimiter=",")
    # plot the graph if required
    if Show == True:
        plt.show()

    return
