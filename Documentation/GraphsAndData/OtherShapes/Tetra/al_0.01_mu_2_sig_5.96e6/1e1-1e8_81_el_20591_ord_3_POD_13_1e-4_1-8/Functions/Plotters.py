#Importing
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

#This is required for when a copy of the file is sent to the results section
try:
    sys.path.insert(0,"Settings")
except:
    pass

from PlotterSettings import PlotterSettings


#Function definition to edit the tick values to a nice format
def TickFormatter(value,tick_number):
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




def PODEigPlotter(savename,Array,PODArray,EigenValues,PODEigenValues):
    #Create a way to reference xkcd colours
    PYCOL=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    
    #Retrieve the settings for the plot
    Title, Show, ETP, _, MLS, MMS, SLS, SMS = PlotterSettings()
    
    #Plot the real graph
    fig, ax = plt.subplots()
    
    #Plot the mainlines
    for i,line in enumerate(ETP):
        if i==0:
            lines = ax.plot(Array,EigenValues[:,line-1].real,MLS,markersize=MMS,color=PYCOL[i])
        else:
            lines += ax.plot(Array,EigenValues[:,line-1].real,MLS,markersize=MMS,color=PYCOL[i])
    
    #Plot the snapshots
    for i,line in enumerate(ETP):
        lines += ax.plot(PODArray,PODEigenValues[:,line-1].real,SLS,markersize=SMS,color=PYCOL[i])
    
    #Format the axes
    plt.xscale('log')
    ax.grid(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(TickFormatter))
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.94, top=0.90)
    
    #Label the axes
    plt.xlabel("Frequency (rad/s)")
    plt.ylabel(r"$\lambda(\mathcal{N}^0+\mathcal{R})$")
    
    #Title
    if Title==True:
        plt.title(r"Eigenvalues of $\mathcal{N}^0+\mathcal{R}$")
    
    #Create the legend
    names = []
    for i,number in enumerate(ETP):
        names.append(r"$\lambda_{"+str(number)+"}(\mathcal{N}^0+\mathcal{R})$ (POD)")
    for i,number in enumerate(ETP):
        names.append(r"$\lambda_{"+str(number)+"}(\mathcal{N}^0+\mathcal{R})$ (Snapshot)")
    
    ax.legend(lines,names)
    
    #Save the graph
    plt.savefig(savename+"RealEigenvalues.pdf")
    
    
    
    #Plot the imaginary graph
    fig, ax = plt.subplots()
    
    #Plot the mainlines
    for i,line in enumerate(ETP):
        if i==0:
            lines = ax.plot(Array,EigenValues[:,line-1].imag,MLS,markersize=MMS,color=PYCOL[i])
        else:
            lines += ax.plot(Array,EigenValues[:,line-1].imag,MLS,markersize=MMS,color=PYCOL[i])
    
    #Plot the snapshots
    for i,line in enumerate(ETP):
        lines += ax.plot(PODArray,PODEigenValues[:,line-1].imag,SLS,markersize=SMS,color=PYCOL[i])
    
    #Format the axes
    plt.xscale('log')
    ax.grid(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(TickFormatter))
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.94, top=0.90)
    
    #Label the axes
    plt.xlabel("Frequency (rad/s)")
    plt.ylabel(r"$\lambda(\mathcal{I})$")
    
    #Title
    if Title==True:
        plt.title(r"Eigenvalues of $\mathcal{I}$")
    
    #Create the legend
    names = []
    for i,number in enumerate(ETP):
        names.append(r"$\lambda_{"+str(number)+"}(\mathcal{I})$ (POD)")
    for i,number in enumerate(ETP):
        names.append(r"$\lambda_{"+str(number)+"}(\mathcal{I})$ (Snapshot)")
    
    ax.legend(lines,names)
    
    #Save the graph
    plt.savefig(savename+"ImaginaryEigenvalues.pdf")
    
    
    return Show


def EigPlotter(savename,Array,EigenValues):
    #Create a way to reference xkcd colours
    PYCOL=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    
    #Retrieve the settings for the plot
    Title, Show, ETP, _, MLS, MMS, _, _ = PlotterSettings()
    
    #Plot the graph
    fig, ax = plt.subplots()
    
    #Plot the mainlines
    for i,line in enumerate(ETP):
        if i==0:
            lines = ax.plot(Array,EigenValues[:,line-1].real,MLS,markersize=MMS,color=PYCOL[i])
        else:
            lines += ax.plot(Array,EigenValues[:,line-1].real,MLS,markersize=MMS,color=PYCOL[i])
    
    #Format the axes
    plt.xscale('log')
    ax.grid(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(TickFormatter))
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.94, top=0.90)
    
    #Label the axes
    plt.xlabel("Frequency (rad/s)")
    plt.ylabel(r"$\lambda(\mathcal{N}^0+\mathcal{R})$")
    
    #Title
    if Title==True:
        plt.title(r"Eigenvalues of $\mathcal{N}^0+\mathcal{R}$")
    
    #Create the legend
    names = []
    for i,number in enumerate(ETP):
        names.append(r"$\lambda_{"+str(number)+"}(\mathcal{N}^0+\mathcal{R})$")
    
    ax.legend(lines,names)
    
    #Save the graph
    plt.savefig(savename+"RealEigenvalues.pdf")
    
    
    #Plot the imaginary graph
    fig, ax = plt.subplots()
    
    #Plot the mainlines
    for i,line in enumerate(ETP):
        if i==0:
            lines = ax.plot(Array,EigenValues[:,line-1].imag,MLS,markersize=MMS,color=PYCOL[i])
        else:
            lines += ax.plot(Array,EigenValues[:,line-1].imag,MLS,markersize=MMS,color=PYCOL[i])
    
    #Format the axes
    plt.xscale('log')
    ax.grid(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(TickFormatter))
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.94, top=0.90)
    
    #Label the axes
    plt.xlabel("Frequency (rad/s)")
    plt.ylabel(r"$\lambda(\mathcal{I})$")
    
    #Title
    if Title==True:
        plt.title(r"Eigenvalues of $\mathcal{I}$")
    
    #Create the legend
    names = []
    for i,number in enumerate(ETP):
        names.append(r"$\lambda_{"+str(number)+"}(\mathcal{I})$")
    
    ax.legend(lines,names)
    
    #Save the graph
    plt.savefig(savename+"ImaginaryEigenvalues.pdf")
    
    
    return Show
    

def PODTensorPlotter(savename,Array,PODArray,Values,PODValues):
    #Create a way to reference xkcd colours
    PYCOL=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    
    #Retrieve the settings for the plot
    Title, Show, _, TTP,MLS, MMS, SLS, SMS = PlotterSettings()
    
    #Plot the graph
    fig, ax = plt.subplots()
    
    #Plot the mainlines
    for i,line in enumerate(TTP):
        if i==0:
            lines = ax.plot(Array,Values[:,line-1].real,MLS,markersize=MMS,color=PYCOL[i])
        else:
            lines += ax.plot(Array,Values[:,line-1].real,MLS,markersize=MMS,color=PYCOL[i])
    
    #Plot the snapshots
    for i,line in enumerate(TTP):
        lines += ax.plot(PODArray,PODValues[:,line-1].real,SLS,markersize=SMS,color=PYCOL[i])
    
    #Format the axes
    plt.xscale('log')
    ax.grid(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(TickFormatter))
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.94, top=0.90)
    
    #Label the axes
    plt.xlabel("Frequency (rad/s)")
    plt.ylabel(r"$\mathcal{N}^0_{ij}+\mathcal{R}_{ij}$")
    
    #Title
    if Title==True:
        plt.title(r"Tensor coefficients of $\mathcal{N}^0+\mathcal{R}$")
    
    #Create the legend
    names = []
    CoefficientRef = ["11","12","13","22","23","33","21","31","_","32"]
    for i,number in enumerate(TTP):
        if number == 1 or number == 4 or number == 6:
            names.append(r"Re($\mathcal{M}_{"+CoefficientRef[number-1]+"}(\omega)$) (POD)")
        else:
            names.append(r"Re($\mathcal{M}_{"+CoefficientRef[number-1]+"}(\omega)$)=Im($\mathcal{M}_{"+CoefficientRef[number+4]+"}(\omega)$) (POD)")
    for i,number in enumerate(TTP):
        if number == 1 or number == 4 or number == 6:
            names.append(r"Re($\mathcal{M}_{"+CoefficientRef[number-1]+"}(\omega)$) (Snapshot)")
        else:
            names.append(r"Re($\mathcal{M}_{"+CoefficientRef[number-1]+"}(\omega)$)=Im($\mathcal{M}_{"+CoefficientRef[number+4]+"}(\omega)$) (Snapshot)")
    #Shrink the size of the legend if there are to many lines
    if len(names)>6:
        ax.legend(lines,names,prop={'size':8})
    else:
        ax.legend(lines,names)

    #Save the graph
    plt.savefig(savename+"RealTensorCoeficients.pdf")
    
    
    #Plot the imaginary graph
    fig, ax = plt.subplots()
    
    #Plot the mainlines
    for i,line in enumerate(TTP):
        if i==0:
            lines = ax.plot(Array,Values[:,line-1].imag,MLS,markersize=MMS,color=PYCOL[i])
        else:
            lines += ax.plot(Array,Values[:,line-1].imag,MLS,markersize=MMS,color=PYCOL[i])
    
    #Plot the snapshots
    for i,line in enumerate(TTP):
        lines += ax.plot(PODArray,PODValues[:,line-1].imag,SLS,markersize=SMS,color=PYCOL[i])
    
    #Format the axes
    plt.xscale('log')
    ax.grid(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(TickFormatter))
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.94, top=0.90)
    
    #Label the axes
    plt.xlabel("Frequency (rad/s)")
    plt.ylabel(r"$\mathcal{I}_{ij}$")
    
    if Title==True:
        plt.title(r"Tensor coefficients of $\mathcal{I}$")
    
    #Create the legend
    names = []
    for i,number in enumerate(TTP):
        if number == 1 or number == 4 or number == 6:
            names.append(r"Im($\mathcal{M}_{"+CoefficientRef[number-1]+"}(\omega)$) (POD)")
        else:
            names.append(r"Im($\mathcal{M}_{"+CoefficientRef[number-1]+"}(\omega)$)=Re($\mathcal{M}_{"+CoefficientRef[number+4]+"}(\omega)$) (POD)")
    for i,number in enumerate(TTP):
        if number == 1 or number == 4 or number == 6:
            names.append(r"Im($\mathcal{M}_{"+CoefficientRef[number-1]+"}(\omega)$) (Snapshot)")
        else:
            names.append(r"Im($\mathcal{M}_{"+CoefficientRef[number-1]+"}(\omega)$)=Re($\mathcal{M}_{"+CoefficientRef[number+4]+"}(\omega)$) (Snapshot)")
    #Shrink the size of the legend if there are to many lines
    if len(names)>6:
        ax.legend(lines,names,prop={'size':8})
    else:
        ax.legend(lines,names)
    
    #Save the graph
    plt.savefig(savename+"ImaginaryTensorCoeficients.pdf")

    
    return Show



def TensorPlotter(savename,Array,Values):
    #Create a way to reference xkcd colours
    PYCOL=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    
    #Retrieve the settings for the plot
    Title, Show, _, TTP, MLS, MMS, _, _ = PlotterSettings()
    
    #Plot the graph
    fig, ax = plt.subplots()
    
    #Plot the mainlines
    for i,line in enumerate(TTP):
        if i==0:
            lines = ax.plot(Array,Values[:,line-1].real,MLS,markersize=MMS,color=PYCOL[i])
        else:
            lines += ax.plot(Array,Values[:,line-1].real,MLS,markersize=MMS,color=PYCOL[i])
    
    #Format the axes
    plt.xscale('log')
    ax.grid(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(TickFormatter))
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.94, top=0.90)
    
    #Label the axes
    plt.xlabel("Frequency (rad/s)")
    plt.ylabel(r"$\mathcal{N}^0_{ij}+\mathcal{R}_{ij}$")
    
    if Title==True:
        plt.title(r"Tensor coefficients of $\mathcal{N}^0+\mathcal{R}$")
    
    #Create the legend
    names = []
    CoefficientRef = ["11","12","13","22","23","33","21","31","_","32"]
    for i,number in enumerate(TTP):
        if number == 1 or number == 4 or number == 6:
            names.append(r"Re($\mathcal{M}_{"+CoefficientRef[number-1]+"}(\omega)$)")
        else:
            names.append(r"Re($\mathcal{M}_{"+CoefficientRef[number-1]+"}(\omega)$)=Re($\mathcal{M}_{"+CoefficientRef[number+4]+"}(\omega)$)")
    
    ax.legend(lines,names)
    
    #Save the graph
    plt.savefig(savename+"RealTensorCoeficients.pdf")
    
    
    #Plot the imaginary graph
    fig, ax = plt.subplots()
    
    #Plot the mainlines
    for i,line in enumerate(TTP):
        if i==0:
            lines = ax.plot(Array,Values[:,line-1].imag,MLS,markersize=MMS,color=PYCOL[i])
        else:
            lines += ax.plot(Array,Values[:,line-1].imag,MLS,markersize=MMS,color=PYCOL[i])
    
    #Format the axes
    plt.xscale('log')
    ax.grid(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(TickFormatter))
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.94, top=0.90)
    
    #Label the axes
    plt.xlabel("Frequency (rad/s)")
    plt.ylabel(r"$\mathcal{I}_{ij}$")
    
    if Title==True:
        plt.title(r"Tensor coefficients of $\mathcal{I}$")
    
    #Create the legend
    names = []
    for i,number in enumerate(TTP):
        if number == 1 or number == 4 or number == 6:
            names.append(r"Im($\mathcal{M}_{"+CoefficientRef[number-1]+"}(\omega)$)")
        else:
            names.append(r"Im($\mathcal{M}_{"+CoefficientRef[number-1]+"}(\omega)$)=Im($\mathcal{M}_{"+CoefficientRef[number+4]+"}(\omega)$)")
    
    ax.legend(lines,names)
    
    #Save the graph
    plt.savefig(savename+"ImaginaryTensorCoeficients.pdf")

    
    return Show

