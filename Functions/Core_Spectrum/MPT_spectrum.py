
import os
import sys
import time
import multiprocessing as multiprocessing
#multiprocessing.set_start_method("spawn", force=True) # ADDED LINE
import tqdm.auto as tqdm
import cmath
import numpy as np

import netgen.meshing as ngmeshing
from ngsolve import *

from ..Core_MPT.MPT_Preallocation import *


def MPT_spectrum(evals, evecs, Theta0Sol, Theta0i, Theta0j, fes, Omega, alpha, sigma_avg, Object, Order, inorout, mur, sig, sweepname, drop_tol, N0, Minf, curve=5, num_solver_threads='default'):

    _, Mu0, _, _, _, _,_, inout, mesh, mu_inv, numelements, sigma, bilinear_bonus_int_order = MPT_Preallocation([Omega], Object, [], curve, inorout,
                                                                                                                  mur, sig, Order, 0, sweepname,
                                                                                                                  num_solver_threads, drop_tol)


    print(Omega)
    xivec = [CoefficientFunction((0, -z, y)), CoefficientFunction((z, 0, -x)), CoefficientFunction((-y, x, 0))]

    # Compute sigma average
    sigma_avg=Integrate(inout*sigma,mesh)/Integrate(inout,mesh)
    print("sigma_avg",sigma_avg)

    u = fes.TrialFunction()
    v = fes.TestFunction()

    gfu = GridFunction(fes)#, multidim=len(evecs))
    xigf = GridFunction(fes)

    sMPT = np.zeros((len(Omega),3,3),dtype=complex)
    TensorArray = np.zeros((len(Omega),9),dtype=complex)
    EigenValues=np.zeros((len(Omega),3),dtype=complex)
    # Get A^(k)
    Apt=np.zeros((len(evals)*3,3,3),dtype=complex)

    m = BilinearForm(fes,symmetric=True)#,condense=False)
    m += SymbolicBFI(inout*(sigma/sigma_avg)*u*v,bonus_intorder=bilinear_bonus_int_order)
    with TaskManager():
        m.Assemble()

    nfound=0
    evalsout=np.zeros(3*len(evals))
    #evalsout[0]=evals[0]
    Tol=1e-2
    #Tol=0.
    e_mass = gfu.vec.CreateVector()
    with TaskManager():
        for k in range(len(evals)):

            # eval - evec pairs are not associated with a particular direction
            # (for the case of a sphere only!!!)
            if k>0:
                if np.abs(evals[k-1]-evals[k])/np.abs(evals[k]) > Tol:
                    nfound+=1
            evalsout[nfound]=evals[k]
            gfu.vec.data = evecs[k]


            for i in range(3):
                Theta0i.vec.FV().NumPy()[:] = Theta0Sol[:, i]
                # diagonal part
                #I1=Integrate(inout*gfu*(Theta0i+xivec[i]),mesh)
                xigf.Set(xivec[i])
                e_mass.data= m.mat * (Theta0i.vec+ xigf.vec)
                I1=InnerProduct(e_mass, gfu.vec)
                #I1=Integrate(gfu*(Theta0i+xivec[i]),mesh,definedon=mesh.Materials("object"))
                #print(I1,I1new)
                Apt[nfound,i,i]+=-evals[k]*alpha**3/4.*I1**2

                for j in range(i+1, 3):

                    Theta0j.vec.FV().NumPy()[:] = Theta0Sol[:, j]
                    xigf.Set(xivec[j])
                    e_mass.data= m.mat * (Theta0j.vec+ xigf.vec)
                    I2=InnerProduct(e_mass, gfu.vec)
                    #I2=Integrate(gfu*(Theta0j+xivec[j]),mesh,definedon=mesh.Materials("object"))
                    # Off diagonals
                    Apt[nfound,i,j]+=-evals[k]*alpha**3/4.*I1*I2
                    Apt[nfound,j,i]+=-evals[k]*alpha**3/4.*I1*I2
            #nfound+=1


    #print(evals,evalsout,nfound)
    #nfound-=1

    n=0
    for omega in Omega:
        sMPT[n,:,:]=N0[:,:]
        s=1j*omega
        mu0=4*pi*1e-7
        for k in range(nfound):#(len(evecs)):
            sk=-evalsout[k]/(mu0*sigma_avg*alpha**2) # sk is negative
            sMPT[n,:,:]+=Apt[k,:,:]*(s/(s-sk))

        TensorArray[n,0:9] =sMPT[n,0:3,0:3].flatten() # convert 3 x 3 to list of 9 numbers
        R = sMPT[n, :,:].real # Note this is Rtilde as it include N0
        I = sMPT[n, :,:].imag
        EigenValues[n, :] = np.sort(np.linalg.eigvals(R)) + 1j * np.sort(np.linalg.eigvals(I))
        n+=1

    xi=np.zeros(nfound+1)
    # Add dummy mode corresponding to xi=0
    xi[0]=0.

    for i in range(nfound):
        xi[i+1]=evalsout[i]/(Mu0*sigma_avg*alpha**2) # xi is positive
    # This would be the full ck amplitudes as 3 x 3 tensor
    #ck=-Apt/(2*np.pi*alpha**3)
    # We choose to ouput ck amplitudes as eigenvalue based on Minf eigenvectors
    Minfeig,Q=np.linalg.eig(Minf)
    Ak=np.zeros((3,3),dtype=complex)
    ck=np.zeros((nfound+1,3),dtype=float)
    # Set first (zero mode) to be eigenvalues of N0 (as arranged using eigenvectors of Minf)
    ck[0,:]=np.diag(np.transpose(Q)@N0@Q)
    for i in range(nfound):
        Ak[:,]=Apt[i,:,:]
        # We choose not to scale here and allow standard or user defined scaling in output
        ck[i+1,:]=-np.diag(np.transpose(Q)@Ak@Q)#/(2*np.pi*alpha**3)


    # Note the computed eigenvalues and TensorArray from above will be the complex conjugate of that from the standard MPT-calculator
    # so take conjugate to return the same Format
    TensorArray=np.conj(TensorArray)
    EigenValues=np.conj(EigenValues)


    return xi, ck, TensorArray, EigenValues
