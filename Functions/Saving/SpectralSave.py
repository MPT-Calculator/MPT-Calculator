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
from .EigPlotter import *
from .TensorPlotter import *
from .ErrorPlotter import *
from .FtoS import *
from .DictionaryList import *


def SpectralSave(Geometry, Array, TensorArray, EigenValues, N0, Minf, Pod, PODArray, PODTol, elements, alpha, Order, MeshSize,
             mur, sig, ck, xi, Time, Sgn_step, Sgn_impulse, Amp_scale, inorout, Nfound):
    """
    P.D. Ledger.2026.
    Save data and make folder structure.

    Args:
        Geometry (str): geometry file name. E.g. 'sphere'
        Array (list): array of frequencies in sweep
        TensorArray (np.ndarray): Nx9 complex tensor coefficients.
        EigenValues (np.ndarray): Nx3 complex eigenvalues
        N0 (np.ndarray): 3x3 N0 coefficient
        Minf (np.ndarray): 3x3 Minf coefficient
        Pod (bool): bool for if sweep used POD
        PODArray (list | np.ndarray): list of K frequencies (rad/s) for POD snapshots.
        PODTol (float): Tolerance for truncated SVD
        elements (int): number of elements in mesh
        alpha (float): object size scaling
        Order (int): order of finite element space.
        MeshSize (float): No longer used. Originally this was max element size.
        mur (dict): dictionary of mur in each region
        sig (dict): dictionary of sigma in each region
    """


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
        if Pod == True:
            subfolder2 = strminF + "-" + strmaxF + "_" + str(Points) + "_el_" + str(elements) + "_ord_" + str(
                Order) + "_POD_" + str(PODPoints) + "_" + strPODTol
        else:
            subfolder2 = strminF + "-" + strmaxF + "_" + str(Points) + "_el_" + str(elements) + "_ord_" + str(Order)
        sweepname = objname + "/" + subfolder1 + "/" + subfolder2
    else:
        sweepname = FolderStructure

    # Save the data
    np.savetxt("Results/" + sweepname + "/Data/Amplitudes.csv", ck, delimiter=",")
    np.savetxt("Results/" + sweepname + "/Data/Modes.csv", xi, delimiter=",")
    np.savetxt("Results/" + sweepname + "/Data/Time.csv", Time, delimiter=",")
    np.savetxt("Results/" + sweepname + "/Data/Sgn_impulse.csv", Sgn_impulse, delimiter=",")
    np.savetxt("Results/" + sweepname + "/Data/Sgn_step.csv", Sgn_step, delimiter=",")


    # plot out amplitudes and time_decays
    if Amp_scale == "default":
        vol = Integrate(inorout,mesh)
        Amp_scale = alpha**3*vol
    # Otherwise allow user defined Amp_scale

    savename = "Results/" + sweepname + "/Graphs/"

    # Plot out Amplitudes
    col=['r','b','x']

    for i in range(3):
        plt.figure()
        nfound=Nfound[i]
        ckp=ck[0:nfound,i]
        xip=xi[0:nfound,i]
        plt.stem(np.log10(np.abs(xip[1:])), ckp[1:]/Amp_scale,col[0],label=r"Approximate, $i=$"+str(i+1))
        plt.xlabel(r'$log_{10}(\xi_k)$')
        plt.ylabel(r'$\tilde{c}_{k,i}/Time_{scale}$')
        plt.legend()
        plt.grid()
        plt.savefig(savename+"Amplitudes+Modes"+str(i+1)+".pdf")

    plt.figure()

    # Plot out step response
    for i in range(3):
        #scale=Sgn_step[0,i]
        plt.semilogx(Time,Sgn_step[:,i]/Amp_scale,"r--",label="Numerical step, $\lambda_i$, $i$="+str(i+1))

    plt.legend()
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"$\lambda_i/Time_{scale}$")
    plt.grid(True, which="both")
    plt.savefig(savename+"step_loglog.pdf")

    plt.figure()
    for i in range(3):
        plt.semilogx(Time,Sgn_step[:,i]/Amp_scale,"r--",label="Numerical step, $\lambda_i$, $i$="+str(i+1))

    plt.legend()
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"$\lambda_i/Time_{scale}$")
    plt.grid(True, which="both")
    plt.savefig(savename+"step_semilog.pdf")

    # Impulse response

    plt.figure()

    for i in range(3):
        scale=Sgn_impulse[0,i]
        plt.loglog(Time,Sgn_impulse[:,i]/scale,"r--",label=r"Numerical impulse, $\lambda_i$, $i$="+str(i+1))

    plt.legend()
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"Relative Response")
    plt.grid(True, which="both")
    plt.savefig(savename+"impulse_loglog.pdf")

    plt.figure()
    for i in range(3):
        scale=Sgn_impulse[0,i]
        plt.semilogx(Time,Sgn_impulse[:,i]/scale,"r--",label=r"Numerical impulse, $\lambda_i$, i="+str(i+1))

    plt.legend()
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"Relative Response")
    plt.grid(True, which="both")
    plt.savefig(savename+"impulse_semilog.pdf")
    plt.show()



    return
