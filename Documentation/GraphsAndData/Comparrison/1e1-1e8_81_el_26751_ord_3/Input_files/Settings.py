#This file contains functions which allow the user to edit default parameters
#Importing
import numpy as np
from ngsolve import *


#Function definition to set up default settings 
def DefaultSettings():
    #How many cores to be used (monitor memory consuption)
    CPUs = 4
    #(int)
    
    #How many snapshots should be taken
    PODPoints = 13
    #(int)
    
    #Tolerance to be used in the TSVD
    PODTol = 10**-4
    #(float)
    return CPUs,PODPoints,PODTol
    


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
    epsi = 10**-12
    #(float) regularisation to be used in the problem
    
    #Maximum iterations to be used in solving the problem
    Maxsteps = 5000
    #(int) maximum number of iterations to be used in solving the problem
    #the bddc will converge in most cases in less than 200 iterations
    #the local will take more
    
    #Relative tolerance
    Tolerance = 10**-10
    #(float) the amount the redsidual must decrease by relatively to solve
    #the problem
    
    #print convergence of the problem
    ngsglobals.msg_level = 0
    #(int) Do you want information about the solving of the problems
    #Suggested inputs
    #0 for no information, 3 for information of convergence
    #Other useful options 1,6
    return Solver,epsi,Maxsteps,Tolerance
