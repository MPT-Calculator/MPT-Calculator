#This file contains functions which allow the user to edit default parameters
#Importing
import numpy as np
from ngsolve import *

# from ngsolve import ngsglobals
# ngsglobals.msg_level = 0

#Function definition to set up default settings 
def DefaultSettings():
    #How many cores to be used (monitor memory consuption)
    CPUs = 4
    #(int)
    
    #Is it a big problem (more memory efficiency but slower)
    BigProblem = False
    #(boolean)
    
    #How many snapshots should be taken
    PODPoints = 13
    #(int)
    
    #Tolerance to be used in the TSVD
    PODTol = 10**-6
    #(float)
    
    #Use an old mesh
    OldMesh = False
    #(boolean) Note that this still requires the relavent .geo file to obtain
    #information about the materials in the mesh
    
    return CPUs,BigProblem,PODPoints,PODTol,OldMesh

def AdditionalOutputs():
    #Plot the POD points
    PlotPod = True
    #(boolean) do you want to plot the snapshots (This requires additional
    #calculations and will slow down sweep by around 2% for default settings)
    
    #Produce certificate bounds for POD outputs
    PODErrorBars = True
    #(boolean)
    
    #Test where the eddy-current model breaks for the object
    EddyCurrentTest = False
    #(boolean)
    
    #Produce a vtk outputfile for the eddy-currents (outputs a large file!)
    vtk_output = False
    #(boolean) do you want to produce a vtk file of the eddy currents in the
    #object (single frequency only)
    
    #Refine the vtk output (extremely large file!)
    Refine_vtk = False
    #(boolean) do you want ngsolve to refine the solution before exporting
    #to the vtk file (single frequency only)
    #(not compatable with all NGSolve versions)
    
    return PlotPod, PODErrorBars, EddyCurrentTest, vtk_output, Refine_vtk
    


#Function definition to set up default settings 
def SaverSettings():
    #Place to save the results to
    FolderName = "Default"
    #(string) This defines the folder (and potentially subfolders) the
    #data will be saved in (if "Default" then a predetermined the data
    #will be saved in a predetermined folder structure)
    #Example input "MyShape/MyFrequencySweep"

    return FolderName



#Function definition to set up parameters relating to solving the problems
def SolverParameters():
    #Parameters associated with solving the problem can edit this
    #preconditioner to be used
    Solver = "bddc"
    #(string) "bddc"/"local"
    
    #regularisation
    epsi = 10**-9
    #(float) regularisation to be used in the problem
    
    #Maximum iterations to be used in solving the problem
    Maxsteps = 2500
    #(int) maximum number of iterations to be used in solving the problem
    #the bddc will converge in most cases in less than 200 iterations
    #the local will take more
    
    #Relative tolerance
    Tolerance = 10**-9
    #(float) the amount the redsidual must decrease by relatively to solve
    #the problem
    
    #print convergence of the problem
    ngsglobals.msg_level = 0
    #(int) Do you want information about the solving of the problems
    #Suggested inputs
    #0 for no information, 3 for information of convergence
    #Other useful options 1,6
    return Solver,epsi,Maxsteps,Tolerance
