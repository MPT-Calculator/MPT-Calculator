#Importing
import math
import cmath
import numpy as np
import csv
from PlotterSaver import MultiPlotterSaver


def exact(Omega):
    Alpha = 0.01
    Mur = 1.5
    Sigma = 6*10**6
    Mu0=4*np.pi*10**(-7)
    nu=Omega*Mu0*Sigma*(Alpha**2)
    k=cmath.sqrt(1j*Mur*Mu0*Sigma*Omega)
    v=k*Alpha;
    Im12=cmath.sqrt(2/np.pi/v)*np.cosh(v)
    Ip12=cmath.sqrt(2/np.pi/v)*np.sinh(v)
    m=2*np.pi*(Alpha**3)*( (2*(Mur*Mu0)+(Mu0))*v*Im12-(Mu0*(1+(v**2))+2*(Mur*Mu0))*Ip12)/(((Mur*Mu0)-Mu0)*v*Im12+((Mu0)*(1+(v**2))-(Mur*Mu0))*Ip12)
    m=m-(2*m.imag*1j)
    return m


#Retrieve the MPT
Frequencies = np.genfromtxt("Frequencies.csv",delimiter=",",dtype = float)
TestEigen = np.genfromtxt("FullEigenvalues.csv",delimiter=",",dtype=complex)
TestEigenPOD = np.genfromtxt("PODEigenvalues.csv",delimiter=",",dtype=complex)
ExactEigen = np.zeros([len(Frequencies),1],dtype = complex)
for i in range(len(Frequencies)):
    ExactEigen[i,0] = exact(Frequencies[i])


Plotname = "FullExactReal"
xs = [Frequencies]
ys = np.zeros([len(Frequencies),2])
ys[:,0] = TestEigen[:,0].real
ys[:,1] = ExactEigen[:,0].real
numberoflines = [2]
styles = [['-+','-x']]
colours = [[1,4]]
names = [r'$\lambda_1(\mathcal{N}^0+\mathcal{R})$',r'$\lambda^{exact}_1(\mathcal{N}^0+\mathcal{R})$']
xlabel = r'Frequency (rad/s)'
ylabel = r'$\lambda_1(\mathcal{N}^0+\mathcal{R})$'
xlog = True
ylog = False
title = False
show = False


MultiPlotterSaver(Plotname,xs,ys,numberoflines,styles,colours,names,xlabel,ylabel,xlog,ylog,title,show)



Plotname = "FullExactImag"
xs = [Frequencies]
ys = np.zeros([len(Frequencies),2])
ys[:,0] = TestEigen[:,0].imag
ys[:,1] = ExactEigen[:,0].imag
numberoflines = [2]
styles = [['-+','-x']]
colours = [[1,4]]
names = [r'$\lambda_1(\mathcal{N}^0+\mathcal{R})$',r'$\lambda^{exact}_1(\mathcal{N}^0+\mathcal{R})$']
xlabel = r'Frequency (rad/s)'
ylabel = r'$\lambda_1(\mathcal{N}^0+\mathcal{R})$'
xlog = True
ylog = False
title = False
show = False


MultiPlotterSaver(Plotname,xs,ys,numberoflines,styles,colours,names,xlabel,ylabel,xlog,ylog,title,show)


ErrorEigen = abs(TestEigen-ExactEigen)/abs(ExactEigen)



Plotname = "FullError"
xs = [Frequencies]
ys = ErrorEigen
numberoflines = [1]
styles = [['-x']]
colours = [[1]]
names = ['Relative Error']
xlabel = r'Frequency (rad/s)'
ylabel = r'Relative Error $\frac{|\lambda_1(\mathcal{N}^0+\mathcal{R})+\lambda_1(\mathcal{I})\mathrm{i}-\lambda^{exact}_1(\mathcal{N}^0+\mathcal{R})-\lambda^{exact}_1(\mathcal{I})\mathrm{i}|}{|\lambda^{exact}_1(\mathcal{N}^0+\mathcal{R})+\lambda^{exact}_1(\mathcal{I})\mathrm{i}|}$'
xlog = True
ylog = False
title = False
show = False


MultiPlotterSaver(Plotname,xs,ys,numberoflines,styles,colours,names,xlabel,ylabel,xlog,ylog,title,show)








Plotname = "PODExactReal"
xs = [Frequencies]
ys = np.zeros([len(Frequencies),2])
ys[:,0] = TestEigenPOD[:,0].real
ys[:,1] = ExactEigen[:,0].real
numberoflines = [2]
styles = [['-+','-x']]
colours = [[1,4]]
names = [r'$\lambda_1(\mathcal{N}^0+\mathcal{R})$',r'$\lambda^{exact}_1(\mathcal{N}^0+\mathcal{R})$']
xlabel = r'Frequency (rad/s)'
ylabel = r'$\lambda_1(\mathcal{N}^0+\mathcal{R})$'
xlog = True
ylog = False
title = False
show = False


MultiPlotterSaver(Plotname,xs,ys,numberoflines,styles,colours,names,xlabel,ylabel,xlog,ylog,title,show)



Plotname = "PODExactImag"
xs = [Frequencies]
ys = np.zeros([len(Frequencies),2])
ys[:,0] = TestEigenPOD[:,0].imag
ys[:,1] = ExactEigen[:,0].imag
numberoflines = [2]
styles = [['-+','-x']]
colours = [[1,4]]
names = [r'$\lambda_1(\mathcal{N}^0+\mathcal{R})$',r'$\lambda^{exact}_1(\mathcal{N}^0+\mathcal{R})$']
xlabel = r'Frequency (rad/s)'
ylabel = r'$\lambda_1(\mathcal{N}^0+\mathcal{R})$'
xlog = True
ylog = False
title = False
show = False


MultiPlotterSaver(Plotname,xs,ys,numberoflines,styles,colours,names,xlabel,ylabel,xlog,ylog,title,show)



ErrorEigenPOD = abs(TestEigenPOD-ExactEigen)/abs(ExactEigen)



Plotname = "PODError"
xs = [Frequencies]
ys = ErrorEigenPOD
numberoflines = [1]
styles = [['-x']]
colours = [[1]]
names = ['Relative Error']
xlabel = r'Frequency (rad/s)'
ylabel = r'Relative Error $\frac{|\lambda_1(\mathcal{N}^0+\mathcal{R})+\lambda_1(\mathcal{I})\mathrm{i}-\lambda^{exact}_1(\mathcal{N}^0+\mathcal{R})-\lambda^{exact}_1(\mathcal{I})\mathrm{i}|}{|\lambda^{exact}_1(\mathcal{N}^0+\mathcal{R})+\lambda^{exact}_1(\mathcal{I})\mathrm{i}|}$'
xlog = True
ylog = False
title = False
show = True


MultiPlotterSaver(Plotname,xs,ys,numberoflines,styles,colours,names,xlabel,ylabel,xlog,ylog,title,show)




















