#This file contain the function which relate to saving the results
#Functions -FtoS (function which converts a float to a string which can be used in the folder names)
#          -DictionaryList (function to list the material parameters for the folder names)
#          -SingleSave (function to save the resutls of single frequency sweep)
#          -PODSave (function to save and plot the results of the POD sweep)
#          -Fullsave (function to save and plot the results of the full order sweep)
#          -FolderMaker (function to create the default folder structure)
#Importing
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
from Plotters import *

#Function to edit floats to a nice format
def FtoS(value):
    if value==0:
        newvalue = "0"
    elif value==1:
        newvalue = "1"
    elif value==-1:
        newvalue = "-1"
    else:
        for i in range(100):
            if abs(value)<=1:
                if round(abs(value/10**(-i)),2)>=1:
                    power=-i
                    break
            else:
                if round(abs(value/10**(i)),2)<1:
                    power=i-1
                    break
        newvalue=value/(10**power)
        newvalue=str(round(newvalue,2))
        if newvalue[-1]=="0":
            newvalue=newvalue[:-1]
        if newvalue[-1]=="0":
            newvalue=newvalue[:-1]
        if newvalue[-1]==".":
            newvalue=newvalue[:-1]
        newvalue += "e"+str(power)

    return newvalue
    

def DictionaryList(Dictionary,Float):
    ParameterList=[]
    for key in Dictionary:
        if key!="air":
            if Float==True:
                newval = FtoS(Dictionary[key])
            else:
                newval = str(Dictionary[key])
                if newval[-1]=="0":
                    newval=newval[:-1]
                if newval[-1]==".":
                    newval=newval[:-1]
            ParameterList.append(newval)
    ParameterList = ','.join(ParameterList)
    
    return ParameterList



def SingleSave(Geometry, Omega, MPT, EigenValues, N0, elements, alpha, Order, MeshSize, mur, sig, EddyCurrentTest, invariants):
    
    #Find how the user wants the data to be saved
    FolderStructure = SaverSettings()
    
    if FolderStructure=="Default":
        #Create the file structure
        #Define constants for the folder name
        objname = Geometry[:-4]
        strOmega = FtoS(Omega)
        strmur = DictionaryList(mur,False)
        strsig = DictionaryList(sig,True)
        #Define the main folder structure
        subfolder1 = "al_"+str(alpha)+"_mu_"+strmur+"_sig_"+strsig
        subfolder2 = "om_"+strOmega+"_el_"+str(elements)+"_ord_"+str(Order)
        sweepname = objname+"/"+subfolder1+"/"+subfolder2
    else:
        sweepname = FolderStructure
    
    #Save the data
    np.savetxt("Results/"+sweepname+"/Data/MPT.csv",MPT, delimiter=",")
    np.savetxt("Results/"+sweepname+"/Data/Eigenvalues.csv",EigenValues, delimiter=",")
    np.savetxt("Results/"+sweepname+"/Data/N0.csv",N0, delimiter=",")
    np.savetxt("Results/"+sweepname+"/Data/Invariants.csv", invariants, delimiter=",")
    if isinstance(EddyCurrentTest, float):
        f = open('Results/'+sweepname+'/Data/Eddy-current_breakdown.txt','w+')
        f.write('omega = '+str(round(EddyCurrentTest)))
        f.close()
    
    return



def PODSave(Geometry, Array, TensorArray, EigenValues, N0, PODTensors, PODEigenValues, PODArray, PODTol, elements, alpha, Order, MeshSize, mur, sig, ErrorTensors, EddyCurrentTest, invariants, using_iterative_POD=False):
    
    #Find how the user wants the data to be saved
    FolderStructure = SaverSettings()
    
    if FolderStructure=="Default":
        #Create the file structure
        #Define constants for the folder name
        objname = Geometry[:-4]
        minF = Array[0]
        strminF = FtoS(minF)
        maxF = Array[-1]
        strmaxF = FtoS(maxF)
        Points = len(Array)
        PODPoints = len(PODArray)
        strmur = DictionaryList(mur,False)
        strsig = DictionaryList(sig,True)
        strPODTol = FtoS(PODTol)
    
        #Define the main folder structure
        subfolder1 = "al_"+str(alpha)+"_mu_"+strmur+"_sig_"+strsig
        subfolder2 = strminF+"-"+strmaxF+"_"+str(Points)+"_el_"+str(elements)+"_ord_"+str(Order)+"_POD_"+str(PODPoints)+"_"+strPODTol
        if using_iterative_POD is True:
            subfolder2 += '_Iterative_POD'
        sweepname = objname+"/"+subfolder1+"/"+subfolder2
    else:
        sweepname = FolderStructure
    
    #Save the data
    np.savetxt("Results/"+sweepname+"/Data/Frequencies.csv",Array, delimiter=",")
    np.savetxt("Results/"+sweepname+"/Data/PODFrequencies.csv",PODArray, delimiter=",")
    np.savetxt("Results/"+sweepname+"/Data/Eigenvalues.csv",EigenValues, delimiter=",")
    np.savetxt("Results/"+sweepname+"/Data/PODEigenvalues.csv",PODEigenValues, delimiter=",")
    np.savetxt("Results/"+sweepname+"/Data/N0.csv",N0, delimiter=",")
    np.savetxt("Results/"+sweepname+"/Data/Tensors.csv",TensorArray, delimiter=",")
    np.savetxt("Results/"+sweepname+"/Data/PODTensors.csv",PODTensors, delimiter=",")
    np.savetxt("Results/"+sweepname+"/Data/Invariants.csv", invariants, delimiter=",")

    if isinstance(EddyCurrentTest, float):
        f = open('Results/'+sweepname+'/Data/Eddy-current_breakdown.txt','w+')
        f.write('omega = '+str(round(EddyCurrentTest)))
        f.close()
    
    
    #Format the tensor arrays so they can be plotted
    PlottingTensorArray = np.zeros([Points,6],dtype=complex)
    PlottingPODTensors = np.zeros([PODPoints,6],dtype=complex)
    PlottingTensorArray = np.concatenate([np.concatenate([TensorArray[:,:3],TensorArray[:,4:6]],axis=1),TensorArray[:,8:9]],axis=1)
    PlottingPODTensors = np.concatenate([np.concatenate([PODTensors[:,:3],PODTensors[:,4:6]],axis=1),PODTensors[:,8:9]],axis=1)
    try:
        ErrorTensors[:,[1,3]] = ErrorTensors[:,[3,1]]
        ErrorTensors[:,[2,4]] = ErrorTensors[:,[4,2]]
        ErrorTensors[:,[4,5]] = ErrorTensors[:,[5,4]]
    except:
        pass
    
    
    #Define where to save the graphs
    savename = "Results/"+sweepname+"/Graphs/"
    
    #Plot the graphs
    Show = PODEigPlotter(savename,Array,PODArray,EigenValues,PODEigenValues,EddyCurrentTest)
    
    try:
        if ErrorTensors==False:
            Show = PODTensorPlotter(savename,Array,PODArray,PlottingTensorArray,PlottingPODTensors,EddyCurrentTest)
    except:
        Show = PODErrorPlotter(savename,Array,PODArray,PlottingTensorArray,PlottingPODTensors,ErrorTensors,EddyCurrentTest)
        
        #Change the format of the error bars to the format of the Tensors
        Errors = np.zeros([Points,9])
        Errors[:,0] = ErrorTensors[:,0]
        Errors[:,1] = ErrorTensors[:,3]
        Errors[:,2] = ErrorTensors[:,4]
        Errors[:,3] = ErrorTensors[:,3]
        Errors[:,4] = ErrorTensors[:,1]
        Errors[:,5] = ErrorTensors[:,5]
        Errors[:,6] = ErrorTensors[:,4]
        Errors[:,7] = ErrorTensors[:,5]
        Errors[:,8] = ErrorTensors[:,2]
        np.savetxt("Results/"+sweepname+"/Data/ErrorBars.csv",Errors, delimiter=",")
    #plot the graph if required
    if Show==True:
        plt.show()
    
    return


def FullSave(Geometry, Array, TensorArray, EigenValues, N0, Pod, PODArray, PODTol, elements, alpha, Order, MeshSize, mur, sig, ErrorTensors, EddyCurrentTest, invariants):
    
    #Find how the user wants the data to be saved
    FolderStructure = SaverSettings()
    
    if FolderStructure=="Default":
        #Create the file structure
        #Define constants for the folder name
        objname = Geometry[:-4]
        minF = Array[0]
        strminF = FtoS(minF)
        maxF = Array[-1]
        strmaxF = FtoS(maxF)
        Points = len(Array)
        PODPoints = len(PODArray)
        strmur = DictionaryList(mur,False)
        strsig = DictionaryList(sig,True)
        strPODTol = FtoS(PODTol)
    
        #Define the main folder structure
        subfolder1 = "al_"+str(alpha)+"_mu_"+strmur+"_sig_"+strsig
        if Pod==True:
            subfolder2 = strminF+"-"+strmaxF+"_"+str(Points)+"_el_"+str(elements)+"_ord_"+str(Order)+"_POD_"+str(PODPoints)+"_"+strPODTol
        else:
            subfolder2 = strminF+"-"+strmaxF+"_"+str(Points)+"_el_"+str(elements)+"_ord_"+str(Order)
        sweepname = objname+"/"+subfolder1+"/"+subfolder2
    else:
        sweepname = FolderStructure
    
    #Save the data
    np.savetxt("Results/"+sweepname+"/Data/Frequencies.csv",Array, delimiter=",")
    np.savetxt("Results/"+sweepname+"/Data/Eigenvalues.csv",EigenValues, delimiter=",")
    np.savetxt("Results/"+sweepname+"/Data/N0.csv",N0, delimiter=",")
    np.savetxt("Results/"+sweepname+"/Data/Tensors.csv",TensorArray, delimiter=",")
    np.savetxt("Results/"+sweepname+"/Data/Invariants.csv", invariants, delimiter=",")

    if Pod==True:
        np.savetxt("Results/"+sweepname+"/Data/PODFrequencies.csv",PODArray, delimiter=",")
    if isinstance(EddyCurrentTest, float):
        f = open('Results/'+sweepname+'/Data/Eddy-current_breakdown.txt','w+')
        f.write('omega = '+str(round(EddyCurrentTest)))
        f.close()
    
    #Format the tensor arrays so they can be plotted
    PlottingTensorArray = np.zeros([Points,6],dtype=complex)
    PlottingTensorArray = np.concatenate([np.concatenate([TensorArray[:,:3],TensorArray[:,4:6]],axis=1),TensorArray[:,8:9]],axis=1)
    try:
        ErrorTensors[:,[1,3]] = ErrorTensors[:,[3,1]]
        ErrorTensors[:,[2,4]] = ErrorTensors[:,[4,2]]
        ErrorTensors[:,[4,5]] = ErrorTensors[:,[5,4]]
    except:
        pass
    
    #Define where to save the graphs
    savename = "Results/"+sweepname+"/Graphs/"
    
    #Plot the graphs
    Show = EigPlotter(savename,Array,EigenValues,EddyCurrentTest)
    
    if Pod == True:
        try:
            if ErrorTensors==False:
                Show = TensorPlotter(savename,Array,PlottingTensorArray,EddyCurrentTest)
        except:
            Show = ErrorPlotter(savename,Array,PlottingTensorArray,ErrorTensors,EddyCurrentTest)
            #Change the format of the error bars to the format of the Tensors
            Errors = np.zeros([Points,9])
            Errors[:,0] = ErrorTensors[:,0]
            Errors[:,1] = ErrorTensors[:,3]
            Errors[:,2] = ErrorTensors[:,4]
            Errors[:,3] = ErrorTensors[:,3]
            Errors[:,4] = ErrorTensors[:,1]
            Errors[:,5] = ErrorTensors[:,5]
            Errors[:,6] = ErrorTensors[:,4]
            Errors[:,7] = ErrorTensors[:,5]
            Errors[:,8] = ErrorTensors[:,2]
            np.savetxt("Results/"+sweepname+"/Data/ErrorBars.csv",Errors, delimiter=",")
    else:
        Show = TensorPlotter(savename,Array,PlottingTensorArray,EddyCurrentTest)
    
    #plot the graph if required
    if Show==True:
        plt.show()
    
    return
    

def FolderMaker(Geometry, Single, Array, Omega, Pod, PlotPod, PODArray, PODTol, alpha, Order, MeshSize, mur, sig, ErrorTensors, VTK, using_OCC, using_interative_POD=False):
    
    #Find how the user wants the data saved
    FolderStructure = SaverSettings()
    
    if FolderStructure=="Default":
        #Create the file structure
        #Define constants for the folder name
        objname = Geometry[:-4]
        minF = Array[0]
        strminF = FtoS(minF)
        maxF = Array[-1]
        strmaxF = FtoS(maxF)
        stromega = FtoS(Omega)
        Points = len(Array)
        PODPoints = len(PODArray)
        strmur = DictionaryList(mur,False)
        strsig = DictionaryList(sig,True)
        strPODTol = FtoS(PODTol)
        
        #Find out the number of elements in the mesh
        Object = Geometry[:-4]+".vol"

        #Loading the object file
        ngmesh = ngmeshing.Mesh(dim=3)
        ngmesh.Load("VolFiles/"+Object)
    
        #Creating the mesh and defining the element types
        mesh = Mesh("VolFiles/"+Object)
        #mesh.Curve(5)#This can be used to refine the mesh
        elements = mesh.ne
    
        #Define the main folder structure
        subfolder1 = "al_"+str(alpha)+"_mu_"+strmur+"_sig_"+strsig
        if Single==True:
            subfolder2 = "om_"+stromega+"_el_"+str(elements)+"_ord_"+str(Order)
        else:
            if Pod==True:
                subfolder2 = strminF+"-"+strmaxF+"_"+str(Points)+"_el_"+str(elements)+"_ord_"+str(Order)+"_POD_"+str(PODPoints)+"_"+strPODTol
                if using_interative_POD is True:
                    subfolder2 += '_Iterative_POD'
            else:
                subfolder2 = strminF+"-"+strmaxF+"_"+str(Points)+"_el_"+str(elements)+"_ord_"+str(Order)
        sweepname = objname+"/"+subfolder1+"/"+subfolder2
    else:
        sweepname = FolderStructure
    
    if Single==True:
        subfolders = ["Data","Input_files"]
    else:
        subfolders = ["Data","Graphs","Functions","Input_files"]
    
    #Create the folders
    for folder in subfolders:
        try:
            os.makedirs("Results/"+sweepname+"/"+folder)
        except:
            pass
    
    #Create the folders for the VTK output if required
    if VTK == True and Single == True:
        try:
            os.makedirs("Results/vtk_output/"+objname+"/om_"+stromega)
        except:
            pass
        
        #Copy the .geo file to the folder
        copyfile("GeoFiles/"+Geometry,"Results/vtk_output/"+objname+"/om_"+stromega+"/"+Geometry)
    
    #Copy the files required to be able to edit the graphs
    if Single!=True:
        copyfile("Settings/PlotterSettings.py","Results/"+sweepname+"/PlotterSettings.py")
        copyfile("Functions/Plotters.py","Results/"+sweepname+"/Functions/Plotters.py")
        if Pod==True:
            if ErrorTensors==True:
                copyfile("Functions/PlotEditorWithErrorBars.py","Results/"+sweepname+"/PlotEditorWithErrorBars.py")
            if PlotPod==True:
                copyfile("Functions/PODPlotEditor.py","Results/"+sweepname+"/PODPlotEditor.py")
                if ErrorTensors!=False:
                    copyfile("Functions/PODPlotEditorWithErrorBars.py","Results/"+sweepname+"/PODPlotEditorWithErrorBars.py")
        copyfile("Functions/PlotEditor.py","Results/"+sweepname+"/PlotEditor.py")
    copyfile("GeoFiles/"+Geometry,"Results/"+sweepname+"/Input_files/"+Geometry)
    copyfile("Settings/Settings.py","Results/"+sweepname+"/Input_files/Settings.py")
    copyfile("main.py","Results/"+sweepname+"/Input_files/main.py")
    
    #Create a compressed version of the .vol file
    os.chdir('VolFiles')
    zipObj = ZipFile(objname+'.zip','w',ZIP_DEFLATED)
    zipObj.write(Object)
    zipObj.close()
    os.replace(objname+'.zip','../Results/'+sweepname+'/Input_files/'+objname+'.zip')
    os.chdir('..')

    #If using OCC, copy .py file:
    if using_OCC is True:
        copyfile('OCC_Geometry/' + Geometry[:-4]+'.py', "Results/"+sweepname+"/Input_files/"+Geometry[:-4]+'.py')

    return sweepname