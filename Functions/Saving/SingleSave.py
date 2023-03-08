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
from .FtoS import *
from .DictionaryList import *


def SingleSave(Geometry, Omega, MPT, EigenValues, N0, elements, alpha, Order, MeshSize, mur, sig, EddyCurrentTest,
               invariants):
    # Find how the user wants the data to be saved
    FolderStructure = SaverSettings()

    if FolderStructure == "Default":
        # Create the file structure
        # Define constants for the folder name
        objname = Geometry[:-4]
        strOmega = FtoS(Omega)
        strmur = DictionaryList(mur, False)
        strsig = DictionaryList(sig, True)
        # Define the main folder structure
        subfolder1 = "al_" + str(alpha) + "_mu_" + strmur + "_sig_" + strsig
        subfolder2 = "om_" + strOmega + "_el_" + str(elements) + "_ord_" + str(Order)
        sweepname = objname + "/" + subfolder1 + "/" + subfolder2
    else:
        sweepname = FolderStructure

    # Save the data
    np.savetxt("Results/" + sweepname + "/Data/MPT.csv", MPT, delimiter=",")
    np.savetxt("Results/" + sweepname + "/Data/Eigenvalues.csv", EigenValues, delimiter=",")
    np.savetxt("Results/" + sweepname + "/Data/N0.csv", N0, delimiter=",")
    np.savetxt("Results/" + sweepname + "/Data/Invariants.csv", invariants, delimiter=",")
    if isinstance(EddyCurrentTest, float):
        f = open('Results/' + sweepname + '/Data/Eddy-current_breakdown.txt', 'w+')
        f.write('omega = ' + str(round(EddyCurrentTest)))
        f.close()

    return