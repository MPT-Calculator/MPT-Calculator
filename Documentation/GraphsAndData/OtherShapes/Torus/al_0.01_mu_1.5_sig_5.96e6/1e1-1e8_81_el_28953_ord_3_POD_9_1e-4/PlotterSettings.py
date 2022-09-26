#This file contains the function which allows the user to edit plotting settings
#Importing
import numpy as np
from ngsolve import *


#Function definition of the plotter settings
def PlotterSettings():
    
    #Line settings
    EigsToPlot = [1,2,3]
    #(list) Which Eigenvalues should be plotted smallest to largest (this is
    #used for both the main lines and snapshots)
    TensorsToPlot = [1,4,6,2,3,5]
    #(list) Which Tensor coefficients to plot leading diagonals are [1,4,6]
    #and tensor layout can be seen below (this is used for both the main
    #lines and the snapshots) 
    #
    #             (1,2,3)
    # Tensor ref =(_,4,5)
    #             (_,_,6)
    
    #Line styles
    MainLineStyle = '-'
    #(string) Linestyle of the eigenvalue plots (string, see matplotlib for
    #availible linestyles)
    MainMarkerSize = 4
    #(int) markersize of eigenvalue plots (if applicable linestyle is chosen)
    
    #Snapshot styles
    SnapshotLineStyle = 'x'
    #(string) Linestyle of snapshots (if plotted)
    SnapshotMarkerSize = 8
    #(int) markersize of snapshots (if plotted)
    
    #Title
    Title = True
    #(boolean)
    
    #Display graph?
    Show = True
    #(boolean) if false then graph is only saved
    
    
    return Title, Show, EigsToPlot, TensorsToPlot, MainLineStyle, MainMarkerSize, SnapshotLineStyle, SnapshotMarkerSize
