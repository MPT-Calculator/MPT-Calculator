import os
import sys
from math import floor, log10
import numpy as np
from shutil import copyfile, copytree
from zipfile import *

import netgen.meshing as ngmeshing
from ngsolve import Mesh

sys.path.insert(0,"Functions")
from Settings import SaverSettings
from .FtoS import *
from .DictionaryList import *


def FolderMaker(Geometry, Single, Array, Omega, Pod, PlotPod, PODArray, PODTol, alpha, Order, MeshSize, mur, sig,
                ErrorTensors, VTK, using_OCC, using_interative_POD=False):
    # Find how the user wants the data saved
    FolderStructure = SaverSettings()

    if FolderStructure == "Default":
        # Create the file structure
        # Define constants for the folder name
        objname = Geometry[:-4]
        minF = Array[0]
        strminF = FtoS(minF)
        maxF = Array[-1]
        strmaxF = FtoS(maxF)
        stromega = FtoS(Omega)
        Points = len(Array)
        PODPoints = len(PODArray)
        strmur = DictionaryList(mur, False)
        strsig = DictionaryList(sig, True)
        strPODTol = FtoS(PODTol)

        # Find out the number of elements in the mesh
        Object = Geometry[:-4] + ".vol"

        # Loading the object file
        ngmesh = ngmeshing.Mesh(dim=3)
        ngmesh.Load("VolFiles/" + Object)

        # Creating the mesh and defining the element types
        mesh = Mesh("VolFiles/" + Object)
        # mesh.Curve(5)#This can be used to refine the mesh
        elements = mesh.ne

        # Define the main folder structure
        subfolder1 = "al_" + str(alpha) + "_mu_" + strmur + "_sig_" + strsig
        if Single == True:
            subfolder2 = "om_" + stromega + "_el_" + str(elements) + "_ord_" + str(Order)
        else:
            if Pod == True:
                subfolder2 = strminF + "-" + strmaxF + "_" + str(Points) + "_el_" + str(elements) + "_ord_" + str(
                    Order) + "_POD_" + str(PODPoints) + "_" + strPODTol
                if using_interative_POD is True:
                    subfolder2 += '_Iterative_POD'
            else:
                subfolder2 = strminF + "-" + strmaxF + "_" + str(Points) + "_el_" + str(elements) + "_ord_" + str(Order)
        sweepname = objname + "/" + subfolder1 + "/" + subfolder2
    else:
        sweepname = FolderStructure

    if Single == True:
        subfolders = ["Data", "Input_files"]
    else:
        subfolders = ["Data", "Graphs", "Functions", "Input_files"]

    # Create the folders
    for folder in subfolders:
        try:
            os.makedirs("Results/" + sweepname + "/" + folder)
        except:
            pass

    # ### WE HAVE MOVED THE SAVE DIRECTORY FOR THE VTK FILES TO THE OBJECT DATA FOLDER.
    # # Create the folders for the VTK output if required
    # if VTK == True and Single == True:
    #     try:
    #         os.makedirs("Results/vtk_output/" + objname + "/om_" + stromega)
    #     except:
    #         pass
    #
    #     # Copy the .geo file to the folder
    #     copyfile("GeoFiles/" + Geometry, "Results/vtk_output/" + objname + "/om_" + stromega + "/" + Geometry)

    # Copy the files required to be able to edit the graphs
    if Single != True:
        copyfile("Settings/PlotterSettings.py", "Results/" + sweepname + "/PlotterSettings.py")
        copytree("Functions", "Results/" + sweepname +"/Functions", dirs_exist_ok=True)
        if Pod == True:
            if ErrorTensors == True:
                copyfile("Functions/PlotEditorWithErrorBars.py", "Results/" + sweepname + "/PlotEditorWithErrorBars.py")
            if PlotPod == True:
                copyfile("Functions/PODPlotEditor.py", "Results/" + sweepname + "/PODPlotEditor.py")
                if ErrorTensors != False:
                    copyfile("Functions/PODPlotEditorWithErrorBars.py",
                             "Results/" + sweepname + "/PODPlotEditorWithErrorBars.py")
        copyfile("Functions/PlotEditor.py", "Results/" + sweepname + "/PlotEditor.py")
    copyfile("GeoFiles/" + Geometry, "Results/" + sweepname + "/Input_files/" + Geometry)
    copyfile("Settings/Settings.py", "Results/" + sweepname + "/Input_files/Settings.py")
    copyfile("main.py", "Results/" + sweepname + "/Input_files/main.py")

    # Create a compressed version of the .vol file
    os.chdir('VolFiles')
    zipObj = ZipFile(objname + '.zip', 'w', ZIP_DEFLATED)
    zipObj.write(Object)
    zipObj.close()
    os.replace(objname + '.zip', '../Results/' + sweepname + '/Input_files/' + objname + '.zip')
    os.chdir('..')

    # If using OCC, copy .py file:
    if using_OCC is True:
        copyfile('OCC_Geometry/' + Geometry[:-4] + '.py',
                 "Results/" + sweepname + "/Input_files/" + Geometry[:-4] + '.py')

    return sweepname