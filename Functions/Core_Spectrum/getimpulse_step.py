import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.special import erfc
import sympy as sym
from scipy.optimize import fsolve
# This function is to compute the step and impulse response to a conducting object
# from the MPTs spectral signature

# The approach employed follows that described in
# M. Simic, D. Ambrus, V. Bilas, Inversion-Based Magnetic Polarizability Tensor Measurement From Time Domian EMI Data
# IEEE Transactions on Instrumentation and Measurements 72 (2023), 6504211

#   Function to obtain the Impulse and Step Response
def getimpulse_step(MPTinf,N0,time,Poles,Amp,Nfound): #getimpulse_step(Frequencies,Tensors,MPTinf,N0,time):
    # First obtain the Poles and Amplitudes
    #Poles, Amp, Nfound=PolesandAmp(Frequencies,Tensors,MPTinf)

    # Now obtain the responses
    Sgn_impulse=np.zeros((len(time),3),dtype=float)
    #Sgn_step=np.zeros((len(time),3),dtype=float)
    MPTinfeig,Q=np.linalg.eig(MPTinf)#np.linalg.eig(MPTinf))
    # Obtain the eigenvalues (important assumed eigenvectors are shared and use the same eigenvectors as MPTinf for )
    lN0=np.diag(np.transpose(Q)@N0@Q)
    Sgn_step=np.zeros((len(time),3),dtype=float)
    # Note the form of the step response includes a constant term relating to -N0
    for i in range(3):
        Sgn_step[:,i]=-lN0[i]*np.ones(len(time))

    for i in range(3):
        ck=Amp[0:Nfound[i],i]
        xi=Poles[0:Nfound[i],i]
        print(Nfound[i],xi,ck)
        for n in range(1,Nfound[i]):
            Sgn_impulse[:,i]+= ck[n] * xi[n] * np.exp(-time[:] * xi[n])
            Sgn_step[:,i]+= ck[n] * np.exp(-time[:] * xi[n])
        print(Sgn_step[:,i])
    return Sgn_impulse, Sgn_step #Poles, Amp, Nfound, Sgn_impulse, Sgn_step



# Generating the Poles and Amplitudes
def PolesandAmp(Frequencies,Tensors,MPTinf):

    Omega=Frequencies
    N=len(Omega)
#   Put Tensors are stored as N x 9 array, so setup neatly
    MPT=np.zeros((N,3,3),dtype=complex)
    MPTeig=np.zeros((N,3),dtype=complex)
    # Use the same eigenvector matrix just in case the ordering is different.
    MPTinfeig,Q=np.linalg.eig(MPTinf)#np.linalg.eig(MPTinf))

    for n in range(N):
        ten=Tensors[n,:]
        MPT[n,0,0]=ten[0]
        MPT[n,0,1]=ten[1]
        MPT[n,0,2]=ten[2]
        MPT[n,1,0]=ten[3]
        MPT[n,1,1]=ten[4]
        MPT[n,1,2]=ten[5]
        MPT[n,2,0]=ten[6]
        MPT[n,2,1]=ten[7]
        MPT[n,2,2]=ten[8]
        # Obtain the eigenvalues (important assumed eigenvectors are shared and use the same eigenvectors for real and imaginary)
        mpt=MPT[n,:,:]
        reig=np.diag(np.transpose(Q)@np.real(mpt)@Q)
        ieig=np.diag(np.transpose(Q)@np.imag(mpt)@Q)
        MPTeig[n,:]=reig[:]+1j*ieig[:]

    #print(MPTeig)
    #print(MPTinfeig)




    # Obtain the poles and amplitues for each coefficent
    # Generate possible xis
    scaleon=True
    if scaleon == True:
        Scale=np.max(Omega)/2
        OmegaScl=Omega/Scale
    else:
        Scale=1
        OmegaScl=np.copy(Omega)

    Poles=np.zeros((N,3),dtype=float)
    Amp=np.zeros((N,3),dtype=float)
    Nfound=np.zeros((3),dtype=int)
    for i in range(3):
        nfound,amp,poles=get_poles_amp(MPTeig[:,i],MPTinfeig[i],OmegaScl,N,Scale)
        Poles[0:nfound,i]=poles
        Amp[0:nfound,i]=amp
        Nfound[i]=nfound

    return Poles, Amp, Nfound




# This function obtains the poles and amplitudes for a particular coefficent
def get_poles_amp(MPT,MPTinf,OmegaScl,N,Scale):
    # Choose possible relaxation frequencies to be the same as OmegaScl
    Xi=np.copy(OmegaScl)
    K=N
    Z=np.zeros((N,1+K),dtype=complex)
    for n in range(1+K):
        for m in range(N):
            if n==0:
                Z[m,n]=1.
            else:
                Z[m,n]=1./(1.+1j*(OmegaScl[m]/Xi[n-1]))
    ZRe=np.real(Z)
    ZIm=np.imag(Z)
    Ztilde=np.zeros((2*N,1+K))
    for n in range(1+K):
        for m in range(N):
            Ztilde[m,n]=ZRe[m,n]
            Ztilde[m+N,n]=ZIm[m,n]
    #set offset according to M(inf)
    #offset=-np.real(MPT[-1])
    safety=1
    offset=-MPTinf*safety
    # Apply offset to ensure convergence
    hRe=np.real(MPT+offset)
    hIm=np.imag(MPT)

    htilde=np.zeros(2*N)
    for m in range(N):
        htilde[m]=hRe[m]
        htilde[m+N]=hIm[m]

    # Perturb diagaonal elements
    #perturbation=1e-1
    #for k in range(K+1):
#        Ztilde[k,k]+=perturbation

    # Setup rhs to l2 norm of 1
    x=nnls(Ztilde,htilde,atol=1e-17)

    # solution for ck
    c=x[0]
    #print(x)
    # determine xi and c
    #cout=[0,]
    #xiout=[0,]
    #nfound=1
    mysum=0.
    #index=[0]
    cout=[c[0]-offset]
    xiout=[0.]
    nfound=1
    index=[0]
    for n in range(1,1+K):
        if abs(c[n]) > 0:
            #if n==0:
            #    cout.append(c[n])
            #    xiout.append(0)
            #    index.append(n)
            #else:
            #    cout.append(c[n])
            #    xiout.append(Scale*Xi[n-1])
            #    index.append(n)
            cout.append(c[n])
            xiout.append(Scale*Xi[n-1])
            index.append(n)
            nfound+=1
    #print("Found"+str(nfound)+"terms")
    ck=np.array(cout)
    xi=np.array(xiout)
    #print("c="+str(ck))
    #print("xi="+str(xi))
    #print(index)

    if nfound > 1:
        # Post processs and resolve duplicates
        xiout2=np.zeros(nfound)
        cout2=np.zeros(nfound)
        xiout2[0]=xiout[0]
        cout2[0]=cout[0]
        nfoundnew=1
        cnt=1
        xiout2[nfoundnew]=xiout[1]
        cout2[nfoundnew]=cout[1]
        nfoundnew=2
        cnt=2
        for n in range(cnt,nfound):
            if index[n]==index[n-1]+1:
                # duplicate
                # use interpolation
                xiout2[nfoundnew-1]=10**(np.log10(xiout2[nfoundnew-1])+cout[n]/(cout[n]+cout2[nfoundnew-1])*np.log10(xiout[n]/xiout2[nfoundnew-1]))
                cout2[nfoundnew-1]=cout2[nfoundnew-1]+cout[n]

            else:
                cout2[nfoundnew]=cout[n]
                xiout2[nfoundnew]=xiout[n]
                nfoundnew+=1
        nfound=nfoundnew
        xiout=xiout2[0:nfound]
        cout=cout2[0:nfound]

        print("Found"+str(nfound)+"terms")

        ck=np.array(cout)
        xi=np.array(xiout)

        print("c="+str(ck))
        print("xi="+str(xi))
    return nfound,ck,xi
