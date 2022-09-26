#Importing
import numpy as np
import cmath
import csv
from PlotterSaver import MultiPlotterSaver


#To reproduce the graph run it with the following parameters
Plotname = "PODExactImag"
with open("xs.csv", "r") as f:
    reader = csv.reader(f)
    xs = list(reader)
ys = np.genfromtxt("ys.csv",delimiter=",")
numberoflines = np.genfromtxt("linenumbers.csv",delimiter=",")
if np.size(numberoflines)==1:
    numberoflines = [numberoflines]
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
ylabel = "$\lambda_1(\mathcal{N}^0+\mathcal{R})$"
xlog = True
ylog = False
title = False
show = True


MultiPlotterSaver(Plotname,xs,ys,numberoflines,styles,colours,names,xlabel,ylabel,xlog,ylog,title,show)

