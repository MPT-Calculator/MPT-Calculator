#Importing
import numpy as np
import csv
from PlotterSaver import MultiPlotterSaver

ActualReal = np.genfromtxt("SigmaReal/ys.csv",delimiter=",",dtype=float)
ActualImag = np.genfromtxt("SigmaImaginary/ys.csv",delimiter=",",dtype=float)

FullEigs = np.genfromtxt("1e1-1e8_81_el_26751_ord_3/Data/Eigenvalues.csv",delimiter=",",dtype=complex)

PODEigs = np.genfromtxt("1e1-1e8_81_el_26751_ord_3_POD_9_1e-5/Data/Eigenvalues.csv",delimiter=",",dtype=complex)

Frequencies = np.genfromtxt("1e1-1e8_81_el_26751_ord_3/Data/Frequencies.csv",delimiter=",",dtype=float)



#Plot the comparrisons
Plotname = "Real eigenvalues"
xs = [Frequencies]
ys = np.zeros([81,3])
ys[:,0] = ActualReal[:,0]
ys[:,1] = FullEigs[:,0].real
ys[:,2] = PODEigs[:,0].real
numberoflines = [3]
styles = [["-x","-","-"]]
colours = [[1,2,3]]
names = [r"Exact $\lambda_1(\mathcal{N}^0+\mathcal{R})$",r"Full order $\lambda_1(\mathcal{N}^0+\mathcal{R})$",r"ROM (9snapshots) $\lambda_1(\mathcal{N}^0+\mathcal{R})$"]
xlabel = "Frequency (rad/sec)"
ylabel = "$\lambda_1(\mathcal{N}^0+\mathcal{R})$"
xlog = True
ylog = False
title = False
show = False


MultiPlotterSaver(Plotname,xs,ys,numberoflines,styles,colours,names,xlabel,ylabel,xlog,ylog,title,show)



Plotname = "Imaginary eigenvalues"
xs = [Frequencies]
ys = np.zeros([81,3])
ys[:,0] = ActualImag[:,0]
ys[:,1] = FullEigs[:,0].imag
ys[:,2] = PODEigs[:,0].imag
numberoflines = [3]
styles = [["-x","-","-"]]
colours = [[1,2,3]]
names = [r"Exact $\lambda_1(\mathcal{I})$",r"Full order $\lambda_1(\mathcal{I})$",r"ROM (9snapshots) $\lambda_1(\mathcal{I})$"]
xlabel = "Frequency (rad/sec)"
ylabel = "$\lambda_1(\mathcal{I})$"
xlog = True
ylog = False
title = False
show = True




MultiPlotterSaver(Plotname,xs,ys,numberoflines,styles,colours,names,xlabel,ylabel,xlog,ylog,title,show)



