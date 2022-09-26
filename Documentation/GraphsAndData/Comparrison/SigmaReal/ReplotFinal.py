#Importing
import numpy as np
import cmath
import csv
from PlotterSaver import MultiPlotterSaver


#To reproduce the graph run it with the following parameters
Plotname = "SigmaReal"
xs = np.genfromtxt("xs.csv",delimiter=",")
ys = np.genfromtxt("ys.csv",delimiter=",")
numberoflines = np.genfromtxt("linenumbers.csv",delimiter=",")
with open("styles.csv","r") as f:
    reader = csv.reader(f)
    styles = list(reader)
with open("colours.csv","r") as f:
    reader = csv.reader(f)
    colours = list(reader)
with open("names.csv","r") as f:
    reader = csv.reader(f)
    names = list(reader)[0]
xlabel = "Frequency (rad/s)"
ylabel = "$\lambda(\mathcal{N}^0+\mathcal{R})$)"
xlog = True
ylog = False
title = False
show = True


MultiPlotterSaver(Plotname,xs,ys,numberoflines,styles,colours,names,xlabel,ylabel,xlog,ylog,title,show)

