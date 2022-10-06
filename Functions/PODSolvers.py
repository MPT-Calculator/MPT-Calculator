"""
Edit 06 Aug 2022: James Elgy
Changed how N0 was calculated for PODSweep to be consistent with PODSweepMulti.
Changed pool generation to spawn to fix linux bug.

"""
#Importing

import os
import sys
import time
import math
import multiprocessing as multiprocessing

import cmath
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

import netgen.meshing as ngmeshing
from ngsolve import *

sys.path.insert(0,"Functions")
from MPTFunctions import *
sys.path.insert(0,"Settings")
from Settings import SolverParameters

# Importing matplotlib for plotting comparisons
import matplotlib
# matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt



def PODSweep(Object,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,sweepname,SavePOD,PODErrorBars,BigProblem):
    Object = Object[:-4]+".vol"
    #Set up the Solver Parameters
    Solver,epsi,Maxsteps,Tolerance = SolverParameters()
    
    #Loading the object file
    ngmesh = ngmeshing.Mesh(dim=3)
    ngmesh.Load("VolFiles/"+Object)
    
    #Creating the mesh and defining the element types
    mesh = Mesh("VolFiles/"+Object)
    mesh.Curve(5)#This can be used to refine the mesh
    numelements = mesh.ne#Count the number elements
    print(" mesh contains "+str(numelements)+" elements")

    #Set up the coefficients
    #Scalars
    Mu0 = 4*np.pi*10**(-7)
    NumberofSnapshots = len(PODArray)
    NumberofFrequencies = len(Array)
    #Coefficient functions
    mu_coef = [ mur[mat] for mat in mesh.GetMaterials() ]
    mu = CoefficientFunction(mu_coef)
    inout_coef = [inorout[mat] for mat in mesh.GetMaterials() ]
    inout = CoefficientFunction(inout_coef)
    sigma_coef = [sig[mat] for mat in mesh.GetMaterials() ]
    sigma = CoefficientFunction(sigma_coef)
    
    #Set up how the tensor and eigenvalues will be stored
    N0=np.zeros([3,3])
    PODN0Errors = np.zeros([3,1])
    R=np.zeros([3,3])
    I=np.zeros([3,3])
    TensorArray=np.zeros([NumberofFrequencies,9], dtype=complex)
    EigenValues = np.zeros([NumberofFrequencies,3], dtype=complex)



#########################################################################
#Theta0
#This section solves the Theta0 problem to calculate both the inputs for
#the Theta1 problem and calculate the N0 tensor

    #Setup the finite element space
    fes = HCurl(mesh, order=Order, dirichlet="outer", flags = { "nograds" : True })
    #Count the number of degrees of freedom
    ndof = fes.ndof
    
    #Define the vectors for the right hand side
    evec = [ CoefficientFunction( (1,0,0) ), CoefficientFunction( (0,1,0) ), CoefficientFunction( (0,0,1) ) ]
    
    #Setup the grid functions and array which will be used to save
    Theta0i = GridFunction(fes)
    Theta0j = GridFunction(fes)
    Theta0Sol = np.zeros([ndof,3])
    
    
    #Run in three directions and save in an array for later
    for i in range(3):
        Theta0Sol[:,i] = Theta0(fes,Order,alpha,mu,inout,evec[i],Tolerance,Maxsteps,epsi,i+1,Solver)
    print(' solved theta0 problems   ')

    #Calculate the N0 tensor
    VolConstant = Integrate(1-mu**(-1),mesh)
    for i in range(3):
        Theta0i.vec.FV().NumPy()[:] = Theta0Sol[:,i]
        for j in range(3):
            Theta0j.vec.FV().NumPy()[:] = Theta0Sol[:,j]
            if i==j:
                N0[i,j] = (alpha**3)*(VolConstant+(1/4)*(Integrate(mu**(-1)*(InnerProduct(curl(Theta0i),curl(Theta0j))),mesh)))
            else:
                N0[i,j] = (alpha**3/4)*(Integrate(mu**(-1)*(InnerProduct(curl(Theta0i),curl(Theta0j))),mesh))
    
    #Copy the tensor 
    # N0+=np.transpose(N0-np.eye(3)@N0)


#########################################################################
#Theta1
#This section solves the Theta1 problem to calculate the solution vectors
#of the snapshots

    #Setup the finite element space
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes2 = HCurl(mesh, order=Order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    ndof2 = fes2.ndof
    
    #Define the vectors for the right hand side
    xivec = [ CoefficientFunction( (0,-z,y) ), CoefficientFunction( (z,0,-x) ), CoefficientFunction( (-y,x,0) ) ]
    
    if BigProblem == True:
        Theta1Sols = np.zeros([ndof2,NumberofSnapshots,3],dtype=np.complex64)
    else:
        Theta1Sols = np.zeros([ndof2,NumberofSnapshots,3],dtype=complex)
    
    if PlotPod == True:
        PODTensors, PODEigenValues, Theta1Sols[:,:,:] = Theta1_Sweep(PODArray,mesh,fes,fes2,Theta0Sol,xivec,alpha,sigma,mu,inout,Tolerance,Maxsteps,epsi,Solver,N0,NumberofFrequencies,True,True,False,BigProblem)
    else:
        Theta1Sols[:,:,:] = Theta1_Sweep(PODArray,mesh,fes,fes2,Theta0Sol,xivec,alpha,sigma,mu,inout,Tolerance,Maxsteps,epsi,Solver,N0,NumberofFrequencies,True,False,False,BigProblem)
    print(' solved theta1 problems     ')

#########################################################################
#POD

    print(' performing SVD              ',end='\r')
    #Perform SVD on the solution vector matrices
    u1Truncated, s1, vh1 = np.linalg.svd(Theta1Sols[:,:,0], full_matrices=False)
    u2Truncated, s2, vh2 = np.linalg.svd(Theta1Sols[:,:,1], full_matrices=False)
    u3Truncated, s3, vh3 = np.linalg.svd(Theta1Sols[:,:,2], full_matrices=False)
    #Print an update on progress
    print(' SVD complete      ')


    #scale the value of the modes
    s1norm=s1/s1[0]
    s2norm=s2/s2[0]
    s3norm=s3/s3[0]

    #Decide where to truncate
    cutoff=NumberofSnapshots
    for i in range(NumberofSnapshots):
        if s1norm[i]<PODTol:
            if s2norm[i]<PODTol:
                if s3norm[i]<PODTol:
                    cutoff=i
                    break

    #Truncate the SVD matrices
    u1Truncated=u1Truncated[:,:cutoff]
    u2Truncated=u2Truncated[:,:cutoff]
    u3Truncated=u3Truncated[:,:cutoff]


########################################################################
#Create the ROM

    print(' creating reduced order model',end='\r')
    nu_no_omega=Mu0*(alpha**2)
    
    Theta_0=GridFunction(fes)
    u, v = fes2.TnT()
    
    if BigProblem == True:
        a0 = BilinearForm(fes2,symmetric=True)
    else:
        a0 = BilinearForm(fes2)
    a0 += SymbolicBFI((mu**(-1)) * InnerProduct(curl(u),curl(v)))
    a0 += SymbolicBFI((1j) * (1-inout) * epsi * InnerProduct(u,v))
    if BigProblem == True:
        a1 = BilinearForm(fes2,symmetric=True)
    else:
        a1 = BilinearForm(fes2)
    a1 += SymbolicBFI((1j) * inout * nu_no_omega * sigma * InnerProduct(u,v))
    
    a0.Assemble()
    a1.Assemble()
    
    Theta_0.vec.FV().NumPy()[:]=Theta0Sol[:,0]
    r1 = LinearForm(fes2)
    r1 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0,v))
    r1 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[0],v))
    r1.Assemble()
    read_vec = r1.vec.CreateVector()
    write_vec = r1.vec.CreateVector()

    Theta_0.vec.FV().NumPy()[:]=Theta0Sol[:,1]
    r2 = LinearForm(fes2)
    r2 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0,v))
    r2 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[1],v))
    r2.Assemble()

    Theta_0.vec.FV().NumPy()[:]=Theta0Sol[:,2]
    r3 = LinearForm(fes2)
    r3 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0,v))
    r3 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[2],v))
    r3.Assemble()


    # Preallocation
    if PODErrorBars == True:
        fes0 = HCurl(mesh, order=0, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
        ndof0 = fes0.ndof
        RerrorReduced1 = np.zeros([ndof0,cutoff*2+1],dtype=complex)
        RerrorReduced2 = np.zeros([ndof0,cutoff*2+1],dtype=complex)
        RerrorReduced3 = np.zeros([ndof0,cutoff*2+1],dtype=complex)
        ProH = GridFunction(fes2)
        ProL = GridFunction(fes0)
########################################################################
#Create the ROM
    R1 = r1.vec.FV().NumPy()
    R2 = r2.vec.FV().NumPy()
    R3 = r3.vec.FV().NumPy()
    A0H = np.zeros([ndof2,cutoff],dtype=complex)
    A1H = np.zeros([ndof2,cutoff],dtype=complex)
    
#E1
    for i in range(cutoff):
        read_vec.FV().NumPy()[:] = u1Truncated[:,i]
        write_vec.data = a0.mat * read_vec
        A0H[:,i] = write_vec.FV().NumPy()
        write_vec.data = a1.mat * read_vec
        A1H[:,i] = write_vec.FV().NumPy()
    HA0H1 = (np.conjugate(np.transpose(u1Truncated))@A0H)
    HA1H1 = (np.conjugate(np.transpose(u1Truncated))@A1H)
    HR1 = (np.conjugate(np.transpose(u1Truncated))@np.transpose(R1))
    
    if PODErrorBars == True:
        # This constructs W^(i) = [r, A0 U^(m,i), A1 U^(m,i)]. Below eqn 31 in efficient comp paper.
        ProH.vec.FV().NumPy()[:] = R1
        ProL.Set(ProH)
        RerrorReduced1[:,0] = ProL.vec.FV().NumPy()[:]
        for i in range(cutoff):
            ProH.vec.FV().NumPy()[:]=A0H[:,i]
            ProL.Set(ProH)
            RerrorReduced1[:,i+1] = ProL.vec.FV().NumPy()[:]
            ProH.vec.FV().NumPy()[:]=A1H[:,i]
            ProL.Set(ProH)
            RerrorReduced1[:,i+cutoff+1] = ProL.vec.FV().NumPy()[:]
#E2    
    for i in range(cutoff):
        read_vec.FV().NumPy()[:] = u2Truncated[:,i]
        write_vec.data = a0.mat * read_vec
        A0H[:,i] = write_vec.FV().NumPy()
        write_vec.data = a1.mat * read_vec
        A1H[:,i] = write_vec.FV().NumPy()
    HA0H2 = (np.conjugate(np.transpose(u2Truncated))@A0H)
    HA1H2 = (np.conjugate(np.transpose(u2Truncated))@A1H)
    HR2 = (np.conjugate(np.transpose(u2Truncated))@np.transpose(R2))
    
    if PODErrorBars == True:
        ProH.vec.FV().NumPy()[:] = R2
        ProL.Set(ProH)
        RerrorReduced2[:,0] = ProL.vec.FV().NumPy()[:]
        for i in range(cutoff):
            ProH.vec.FV().NumPy()[:]=A0H[:,i]
            ProL.Set(ProH)
            RerrorReduced2[:,i+1] = ProL.vec.FV().NumPy()[:]
            ProH.vec.FV().NumPy()[:]=A1H[:,i]
            ProL.Set(ProH)
            RerrorReduced2[:,i+cutoff+1] = ProL.vec.FV().NumPy()[:]
#E3
    for i in range(cutoff):
        read_vec.FV().NumPy()[:] = u3Truncated[:,i]
        write_vec.data = a0.mat * read_vec
        A0H[:,i] = write_vec.FV().NumPy()
        write_vec.data = a1.mat * read_vec
        A1H[:,i] = write_vec.FV().NumPy()
    HA0H3 = (np.conjugate(np.transpose(u3Truncated))@A0H)
    HA1H3 = (np.conjugate(np.transpose(u3Truncated))@A1H)
    HR3 = (np.conjugate(np.transpose(u3Truncated))@np.transpose(R3))
    
    if PODErrorBars == True:
        ProH.vec.FV().NumPy()[:] = R3
        ProL.Set(ProH)
        RerrorReduced3[:,0] = ProL.vec.FV().NumPy()[:]
        for i in range(cutoff):
            ProH.vec.FV().NumPy()[:]=A0H[:,i]
            ProL.Set(ProH)
            RerrorReduced3[:,i+1] = ProL.vec.FV().NumPy()[:]
            ProH.vec.FV().NumPy()[:]=A1H[:,i]
            ProL.Set(ProH)
            RerrorReduced3[:,i+cutoff+1] = ProL.vec.FV().NumPy()[:]

    #Clear the variables
    A0H, A1H = None, None
    a0, a1 = None, None
    
########################################################################
#Sort out the error bounds
    if PODErrorBars==True:
        if BigProblem == True:
            MR1 = np.zeros([ndof0,cutoff*2+1],dtype=np.complex64)
            MR2 = np.zeros([ndof0,cutoff*2+1],dtype=np.complex64)
            MR3 = np.zeros([ndof0,cutoff*2+1],dtype=np.complex64)
        else:
            MR1 = np.zeros([ndof0,cutoff*2+1],dtype=complex)
            MR2 = np.zeros([ndof0,cutoff*2+1],dtype=complex)
            MR3 = np.zeros([ndof0,cutoff*2+1],dtype=complex)
        
        u, v = fes0.TnT()
        
        m = BilinearForm(fes0)
        m += SymbolicBFI(InnerProduct(u,v))
        f = LinearForm(fes0)
        m.Assemble()
        c = Preconditioner(m,"local")
        c.Update()
        inverse = CGSolver(m.mat,c.mat,precision=1e-20,maxsteps=500)
        
        ErrorGFU = GridFunction(fes0)
        for i in range(2*cutoff+1):
            #E1
            ProL.vec.data.FV().NumPy()[:] = RerrorReduced1[:,i]
            ProL.vec.data -= m.mat * ErrorGFU.vec
            ErrorGFU.vec.data += inverse * ProL.vec
            MR1[:,i] = ErrorGFU.vec.FV().NumPy()
        
            #E2
            ProL.vec.data.FV().NumPy()[:] = RerrorReduced2[:,i]
            ProL.vec.data -= m.mat * ErrorGFU.vec
            ErrorGFU.vec.data += inverse * ProL.vec
            MR2[:,i] = ErrorGFU.vec.FV().NumPy()
        
            #E3
            ProL.vec.data.FV().NumPy()[:] = RerrorReduced3[:,i]
            ProL.vec.data -= m.mat * ErrorGFU.vec
            ErrorGFU.vec.data += inverse * ProL.vec
            MR3[:,i] = ErrorGFU.vec.FV().NumPy()
        
        
        G1 = np.transpose(np.conjugate(RerrorReduced1))@MR1
        G2 = np.transpose(np.conjugate(RerrorReduced2))@MR2
        G3 = np.transpose(np.conjugate(RerrorReduced3))@MR3
        G12 = np.transpose(np.conjugate(RerrorReduced1))@MR2
        G13 = np.transpose(np.conjugate(RerrorReduced1))@MR3
        G23 = np.transpose(np.conjugate(RerrorReduced2))@MR3
        
        #Clear the variables
        # RerrorReduced1, RerrorReduced2, RerrorReduced3 = None, None, None
        # MR1, MR2, MR3 = None, None, None
        # fes0, m, c, inverse = None, None, None, None
        
        
        fes3 = HCurl(mesh, order=Order, dirichlet="outer", gradientdomains=dom_nrs_metal)
        ndof3 = fes3.ndof
        Omega = Array[0]
        u,v = fes3.TnT()
        amax = BilinearForm(fes3)
        amax += (mu**(-1))*curl(u)*curl(v)*dx
        amax += (1-inout)*epsi*u*v*dx
        amax += inout*sigma*(alpha**2)*Mu0*Omega*u*v*dx

        m = BilinearForm(fes3)
        m += u*v*dx

        apre = BilinearForm(fes3)
        apre += curl(u)*curl(v)*dx + u*v*dx
        pre = Preconditioner(amax, "bddc")

        with TaskManager():
            amax.Assemble()
            m.Assemble()
            apre.Assemble()

            # build gradient matrix as sparse matrix (and corresponding scalar FESpace)
            gradmat, fesh1 = fes3.CreateGradient()
            gradmattrans = gradmat.CreateTranspose() # transpose sparse matrix
            math1 = gradmattrans @ m.mat @ gradmat   # multiply matrices
            math1[0,0] += 1     # fix the 1-dim kernel
            invh1 = math1.Inverse(inverse="sparsecholesky")
        
            # build the Poisson projector with operator Algebra:
            proj = IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ m.mat
            projpre = proj @ pre.mat
            evals, evecs = solvers.PINVIT(amax.mat, m.mat, pre=projpre, num=1, maxit=50)

        alphaLB = evals[0]
        print(f'Lower bound alphaLB = {alphaLB} \n')
        #Clear the variables
        fes3, amax, apre, pre, invh1, m = None, None, None, None, None, None




########################################################################
#Produce the sweep using the lower dimensional space
    #Setup variables for calculating tensors
    Theta_0j=GridFunction(fes)
    Theta_1i=GridFunction(fes2)
    Theta_1j=GridFunction(fes2)
    
    if PODErrorBars ==True:
        rom1 = np.zeros([2*cutoff+1,1],dtype=complex)
        rom2 = np.zeros([2*cutoff+1,1],dtype=complex)
        rom3 = np.zeros([2*cutoff+1,1],dtype=complex)
        ErrorTensors=np.zeros([NumberofFrequencies,6])

    # Cleaning up Theta0Sols and Theta1Sols
    # del Theta1Sols
    # del Theta0Sol

    for k,omega in enumerate(Array):

        #This part is for obtaining the solutions in the lower dimensional space
        print(' solving reduced order system %d/%d    ' % (k+1,NumberofFrequencies), end='\r')
        t1 = time.time()
        g1=np.linalg.solve(HA0H1+HA1H1*omega,HR1*omega)
        g2=np.linalg.solve(HA0H2+HA1H2*omega,HR2*omega)
        g3=np.linalg.solve(HA0H3+HA1H3*omega,HR3*omega)
        
        #This part projects the problem to the higher dimensional space
        W1=np.dot(u1Truncated,g1).flatten()
        W2=np.dot(u2Truncated,g2).flatten()
        W3=np.dot(u3Truncated,g3).flatten()
        
        #Calculate the tensors
        nu = omega*Mu0*(alpha**2)
        R=np.zeros([3,3])
        I=np.zeros([3,3])
        
        for i in range(3):
            Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:,i]
            xii = xivec[i]
            if i==0:
                Theta_1i.vec.FV().NumPy()[:]=W1
            if i==1:
                Theta_1i.vec.FV().NumPy()[:]=W2
            if i==2:
                Theta_1i.vec.FV().NumPy()[:]=W3
            for j in range(i+1):
                Theta_0j.vec.FV().NumPy()[:]=Theta0Sol[:,j]
                xij=xivec[j]
                if j==0:
                    Theta_1j.vec.FV().NumPy()[:]=W1
                if j==1:
                    Theta_1j.vec.FV().NumPy()[:]=W2
                if j==2:
                    Theta_1j.vec.FV().NumPy()[:]=W3
                
                #Real and Imaginary parts
                R[i,j]=-(((alpha**3)/4)*Integrate((mu**(-1))*(curl(Theta_1j)*Conj(curl(Theta_1i))),mesh)).real
                I[i,j]=((alpha**3)/4)*Integrate(inout*nu*sigma*((Theta_1j+Theta_0j+xij)*(Conj(Theta_1i)+Theta_0+xii)),mesh).real
        
        R+=np.transpose(R-np.diag(np.diag(R))).real
        I+=np.transpose(I-np.diag(np.diag(I))).real

        #Save in arrays
        TensorArray[k,:] = (N0+R+1j*I).flatten()
        EigenValues[k,:] = np.sort(np.linalg.eigvals(N0+R))+1j*np.sort(np.linalg.eigvals(I))
        
        
        
        if PODErrorBars==True:
            rom1[0,0] = omega
            rom2[0,0] = omega
            rom3[0,0] = omega

            rom1[1:1+cutoff,0] = -g1.flatten()
            rom2[1:1+cutoff,0] = -g2.flatten()
            rom3[1:1+cutoff,0] = -g3.flatten()

            rom1[1+cutoff:,0] = -(g1*omega).flatten()
            rom2[1+cutoff:,0] = -(g2*omega).flatten()
            rom3[1+cutoff:,0] = -(g3*omega).flatten()

            error1 = np.conjugate(np.transpose(rom1))@G1@rom1
            error2 = np.conjugate(np.transpose(rom2))@G2@rom2
            error3 = np.conjugate(np.transpose(rom3))@G3@rom3
            error12 = np.conjugate(np.transpose(rom1))@G12@rom2
            error13 = np.conjugate(np.transpose(rom1))@G13@rom3
            error23 = np.conjugate(np.transpose(rom2))@G23@rom3
    
            error1 = abs(error1)**(1/2)
            error2 = abs(error2)**(1/2)
            error3 = abs(error3)**(1/2)
            error12 = error12.real
            error13 = error13.real
            error23 = error23.real
    
            Errors=[error1,error2,error3,error12,error13,error23]
    
            for j in range(6):
                if j<3:
                    ErrorTensors[k,j] = ((alpha**3)/4)*(Errors[j]**2)/alphaLB
                else:
                    ErrorTensors[k,j] = -2*Errors[j]
                    if j==3:
                        ErrorTensors[k,j] += (Errors[0]**2)+(Errors[1]**2)
                        ErrorTensors[k,j] = ((alpha**3)/(8*alphaLB))*((Errors[0]**2)+(Errors[1]**2)+ErrorTensors[k,j])
                    if j==4:
                        ErrorTensors[k,j] += (Errors[0]**2)+(Errors[2]**2)
                        ErrorTensors[k,j] = ((alpha**3)/(8*alphaLB))*((Errors[0]**2)+(Errors[1]**2)+ErrorTensors[k,j])
                    if j==5:
                        ErrorTensors[k,j] += (Errors[1]**2)+(Errors[2]**2)
                        ErrorTensors[k,j] = ((alpha**3)/(8*alphaLB))*((Errors[0]**2)+(Errors[1]**2)+ErrorTensors[k,j])
    
    #print(ErrorTensors)
    print(' reduced order systems solved        ')
    print(' frequency sweep complete')

    if PlotPod==True:
        if PODErrorBars==True:
            return TensorArray, EigenValues, N0, PODTensors, PODEigenValues, numelements, ErrorTensors, (ndof, ndof2)
        else:
            return TensorArray, EigenValues, N0, PODTensors, PODEigenValues, numelements, (ndof, ndof2)
    else:
        if PODErrorBars==True:
            return TensorArray, EigenValues, N0, numelements, ErrorTensors, (ndof, ndof2)
        else:
            return TensorArray, EigenValues, N0, numelements, (ndof, ndof2)


def PODSweepMulti(Object,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,CPUs,sweepname,SavePOD,PODErrorBars,BigProblem):
    Object = Object[:-4]+".vol"
    #Set up the Solver Parameters
    Solver,epsi,Maxsteps,Tolerance = SolverParameters()
    
    #Loading the object file
    ngmesh = ngmeshing.Mesh(dim=3)
    ngmesh.Load("VolFiles/"+Object)
    
    #Creating the mesh and defining the element types
    mesh = Mesh("VolFiles/"+Object)
    mesh.Curve(5)#This can be used to refine the mesh
    numelements = mesh.ne#Count the number elements
    print(" mesh contains "+str(numelements)+" elements")

    #Set up the coefficients
    #Scalars
    Mu0 = 4*np.pi*10**(-7)
    NumberofSnapshots = len(PODArray)
    NumberofFrequencies = len(Array)
    #Coefficient functions
    mu_coef = [ mur[mat] for mat in mesh.GetMaterials() ]
    mu = CoefficientFunction(mu_coef)
    inout_coef = [inorout[mat] for mat in mesh.GetMaterials() ]
    inout = CoefficientFunction(inout_coef)
    sigma_coef = [sig[mat] for mat in mesh.GetMaterials() ]
    sigma = CoefficientFunction(sigma_coef)
    
    #Set up how the tensor and eigenvalues will be stored
    N0=np.zeros([3,3])
    TensorArray=np.zeros([NumberofFrequencies,9], dtype=complex)
    RealEigenvalues = np.zeros([NumberofFrequencies,3])
    ImaginaryEigenvalues = np.zeros([NumberofFrequencies,3])
    EigenValues = np.zeros([NumberofFrequencies,3], dtype=complex)

#########################################################################
#Theta0
#This section solves the Theta0 problem to calculate both the inputs for
#the Theta1 problem and calculate the N0 tensor

    #Setup the finite element space
    fes = HCurl(mesh, order=Order, dirichlet="outer", flags = { "nograds" : True })
    #Count the number of degrees of freedom
    ndof = fes.ndof
    
    #Define the vectors for the right hand side
    evec = [ CoefficientFunction( (1,0,0) ), CoefficientFunction( (0,1,0) ), CoefficientFunction( (0,0,1) ) ]
    
    #Setup the grid functions and array which will be used to save
    Theta0i = GridFunction(fes)
    Theta0j = GridFunction(fes)
    Theta0Sol = np.zeros([ndof,3])
    
    #Setup the inputs for the functions to run
    Theta0CPUs = min(3,multiprocessing.cpu_count(),CPUs)
    Runlist = []
    for i in range(3):
        if Theta0CPUs<3:
            NewInput = (fes,Order,alpha,mu,inout,evec[i],Tolerance,Maxsteps,epsi,i+1,Solver)
        else:
            NewInput = (fes,Order,alpha,mu,inout,evec[i],Tolerance,Maxsteps,epsi,"No Print",Solver)
        Runlist.append(NewInput)
    #Run on the multiple cores
    with multiprocessing.get_context("spawn").Pool(Theta0CPUs) as pool:
        Output = pool.starmap(Theta0, Runlist)
    print(' solved theta0 problems    ')
    
    #Unpack the outputs
    for i,Direction in enumerate(Output):
        Theta0Sol[:,i] = Direction


#Calculate the N0 tensor
    VolConstant = Integrate(1-mu**(-1),mesh)
    for i in range(3):
        Theta0i.vec.FV().NumPy()[:] = Theta0Sol[:,i]
        for j in range(3):
            Theta0j.vec.FV().NumPy()[:] = Theta0Sol[:,j]
            if i==j:
                N0[i,j] = (alpha**3)*(VolConstant+(1/4)*(Integrate(mu**(-1)*(InnerProduct(curl(Theta0i),curl(Theta0j))),mesh)))
            else:
                N0[i,j] = (alpha**3/4)*(Integrate(mu**(-1)*(InnerProduct(curl(Theta0i),curl(Theta0j))),mesh))



#########################################################################
#Theta1
#This section solves the Theta1 problem and saves the solution vectors

    #Setup the finite element space
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes2 = HCurl(mesh, order=Order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    #Count the number of degrees of freedom
    ndof2 = fes2.ndof
    
    #Define the vectors for the right hand side
    xivec = [ CoefficientFunction( (0,-z,y) ), CoefficientFunction( (z,0,-x) ), CoefficientFunction( (-y,x,0) ) ]



    #Work out where to send each frequency
    Theta1_CPUs = min(NumberofSnapshots,multiprocessing.cpu_count(),CPUs)
    Core_Distribution = []
    Count_Distribution = []
    for i in range(Theta1_CPUs):
        Core_Distribution.append([])
        Count_Distribution.append([])
    
    
    #Distribute between the cores
    CoreNumber = 0
    count = 1
    for i,Omega in enumerate(PODArray):
        Core_Distribution[CoreNumber].append(Omega)
        Count_Distribution[CoreNumber].append(i)
        if CoreNumber == CPUs-1 and count == 1:
            count = -1
        elif CoreNumber == 0 and count == -1:
            count = 1
        else:
            CoreNumber +=count
    
    #Create the inputs
    Runlist = []
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)
    for i in range(Theta1_CPUs):
        if PlotPod == True:
            Runlist.append((Core_Distribution[i],mesh,fes,fes2,Theta0Sol,xivec,alpha,sigma,mu,inout,Tolerance,Maxsteps,epsi,Solver,N0,NumberofSnapshots,True,True,counter,BigProblem))
        else:
            Runlist.append((Core_Distribution[i],mesh,fes,fes2,Theta0Sol,xivec,alpha,sigma,mu,inout,Tolerance,Maxsteps,epsi,Solver,N0,NumberofSnapshots,True,False,counter,BigProblem))
    
    #Run on the multiple cores
    with multiprocessing.get_context("spawn").Pool(Theta1_CPUs) as pool:
        Outputs = pool.starmap(Theta1_Sweep, Runlist)

    try:
        pool.terminate()
        print('manually closed pool')
    except:
        print('Pool has already closed.')
    
    #Unpack the results
    if BigProblem == True:
        Theta1Sols = np.zeros([ndof2,NumberofSnapshots,3],dtype=np.complex64)
    else:
        Theta1Sols = np.zeros([ndof2,NumberofSnapshots,3],dtype=complex)
    if PlotPod == True:
        PODTensors = np.zeros([NumberofSnapshots,9],dtype=complex)
        PODEigenValues = np.zeros([NumberofSnapshots,3],dtype=complex)
    for i,Output in enumerate(Outputs):
        for j,Num in enumerate(Count_Distribution[i]):
            if PlotPod == True:
                PODTensors[Num,:] = Output[0][j]
                PODEigenValues[Num,:] = Output[1][j]
                Theta1Sols[:,Num,:] = Output[2][:,j,:]
            else:
                Theta1Sols[:,Num,:] = Output[:,j,:]
    



#########################################################################
#POD

    print(' performing SVD              ',end='\r')
    #Perform SVD on the solution vector matrices
    u1Truncated, s1, vh1 = np.linalg.svd(Theta1Sols[:,:,0], full_matrices=False)
    u2Truncated, s2, vh2 = np.linalg.svd(Theta1Sols[:,:,1], full_matrices=False)
    u3Truncated, s3, vh3 = np.linalg.svd(Theta1Sols[:,:,2], full_matrices=False)
    #Get rid of the solution vectors
    Theta1Sols = None
    #Print an update on progress
    print(' SVD complete      ')

    #scale the value of the modes
    s1norm=s1/s1[0]
    s2norm=s2/s2[0]
    s3norm=s3/s3[0]

    #Decide where to truncate
    cutoff=NumberofSnapshots
    for i in range(NumberofSnapshots):
        if s1norm[i]<PODTol:
            if s2norm[i]<PODTol:
                if s3norm[i]<PODTol:
                    cutoff=i
                    break

    #Truncate the SVD matrices
    u1Truncated=u1Truncated[:,:cutoff]
    u2Truncated=u2Truncated[:,:cutoff]
    u3Truncated=u3Truncated[:,:cutoff]
    


########################################################################
#Create the ROM

    print(' creating reduced order model',end='\r')
    #Mu0=4*np.pi*10**(-7)
    nu_no_omega=Mu0*(alpha**2)
    
    Theta_0=GridFunction(fes)
    u, v = fes2.TnT()
    
    if BigProblem == True:
        a0 = BilinearForm(fes2,symmetric=True)
    else:
        a0 = BilinearForm(fes2)
    a0 += SymbolicBFI((mu**(-1)) * InnerProduct(curl(u),curl(v)))
    a0 += SymbolicBFI((1j) * (1-inout) * epsi * InnerProduct(u,v))
    if BigProblem == True:
        a1 = BilinearForm(fes2,symmetric=True)
    else:
        a1 = BilinearForm(fes2)
    a1 += SymbolicBFI((1j) * inout * nu_no_omega * sigma * InnerProduct(u,v))
    
    a0.Assemble()
    a1.Assemble()
    
    Theta_0.vec.FV().NumPy()[:]=Theta0Sol[:,0]
    r1 = LinearForm(fes2)
    r1 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0,v))
    r1 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[0],v))
    r1.Assemble()
    read_vec = r1.vec.CreateVector()
    write_vec = r1.vec.CreateVector()

    Theta_0.vec.FV().NumPy()[:]=Theta0Sol[:,1]
    r2 = LinearForm(fes2)
    r2 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0,v))
    r2 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[1],v))
    r2.Assemble()

    Theta_0.vec.FV().NumPy()[:]=Theta0Sol[:,2]
    r3 = LinearForm(fes2)
    r3 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0,v))
    r3 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[2],v))
    r3.Assemble()

    if PODErrorBars == True:
        fes0 = HCurl(mesh, order=0, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
        ndof0 = fes0.ndof
        RerrorReduced1 = np.zeros([ndof0,cutoff*2+1],dtype=complex)
        RerrorReduced2 = np.zeros([ndof0,cutoff*2+1],dtype=complex)
        RerrorReduced3 = np.zeros([ndof0,cutoff*2+1],dtype=complex)
        ProH = GridFunction(fes2)
        ProL = GridFunction(fes0)
########################################################################
#Create the ROM
    R1 = r1.vec.FV().NumPy()
    R2 = r2.vec.FV().NumPy()
    R3 = r3.vec.FV().NumPy()
    A0H = np.zeros([ndof2,cutoff],dtype=complex)
    A1H = np.zeros([ndof2,cutoff],dtype=complex)
    
    
#E1
    for i in range(cutoff):
        read_vec.FV().NumPy()[:] = u1Truncated[:,i]
        write_vec.data = a0.mat * read_vec
        A0H[:,i] = write_vec.FV().NumPy()
        write_vec.data = a1.mat * read_vec
        A1H[:,i] = write_vec.FV().NumPy()
    HA0H1 = (np.conjugate(np.transpose(u1Truncated))@A0H)
    HA1H1 = (np.conjugate(np.transpose(u1Truncated))@A1H)
    HR1 = (np.conjugate(np.transpose(u1Truncated))@np.transpose(R1))
    
    if PODErrorBars == True:
        ProH.vec.FV().NumPy()[:] = R1
        ProL.Set(ProH)
        RerrorReduced1[:,0] = ProL.vec.FV().NumPy()[:]
        for i in range(cutoff):
            ProH.vec.FV().NumPy()[:]=A0H[:,i]
            ProL.Set(ProH)
            RerrorReduced1[:,i+1] = ProL.vec.FV().NumPy()[:]
            ProH.vec.FV().NumPy()[:]=A1H[:,i]
            ProL.Set(ProH)
            RerrorReduced1[:,i+cutoff+1] = ProL.vec.FV().NumPy()[:]
#E2    
    for i in range(cutoff):
        read_vec.FV().NumPy()[:] = u2Truncated[:,i]
        write_vec.data = a0.mat * read_vec
        A0H[:,i] = write_vec.FV().NumPy()
        write_vec.data = a1.mat * read_vec
        A1H[:,i] = write_vec.FV().NumPy()
    HA0H2 = (np.conjugate(np.transpose(u2Truncated))@A0H)
    HA1H2 = (np.conjugate(np.transpose(u2Truncated))@A1H)
    HR2 = (np.conjugate(np.transpose(u2Truncated))@np.transpose(R2))
    
    if PODErrorBars == True:
        ProH.vec.FV().NumPy()[:] = R2
        ProL.Set(ProH)
        RerrorReduced2[:,0] = ProL.vec.FV().NumPy()[:]
        for i in range(cutoff):
            ProH.vec.FV().NumPy()[:]=A0H[:,i]
            ProL.Set(ProH)
            RerrorReduced2[:,i+1] = ProL.vec.FV().NumPy()[:]
            ProH.vec.FV().NumPy()[:]=A1H[:,i]
            ProL.Set(ProH)
            RerrorReduced2[:,i+cutoff+1] = ProL.vec.FV().NumPy()[:]
#E3
    for i in range(cutoff):
        read_vec.FV().NumPy()[:] = u3Truncated[:,i]
        write_vec.data = a0.mat * read_vec
        A0H[:,i] = write_vec.FV().NumPy()
        write_vec.data = a1.mat * read_vec
        A1H[:,i] = write_vec.FV().NumPy()
    HA0H3 = (np.conjugate(np.transpose(u3Truncated))@A0H)
    HA1H3 = (np.conjugate(np.transpose(u3Truncated))@A1H)
    HR3 = (np.conjugate(np.transpose(u3Truncated))@np.transpose(R3))
    
    if PODErrorBars == True:
        ProH.vec.FV().NumPy()[:] = R3
        ProL.Set(ProH)
        RerrorReduced3[:,0] = ProL.vec.FV().NumPy()[:]
        for i in range(cutoff):
            ProH.vec.FV().NumPy()[:]=A0H[:,i]
            ProL.Set(ProH)
            RerrorReduced3[:,i+1] = ProL.vec.FV().NumPy()[:]
            ProH.vec.FV().NumPy()[:]=A1H[:,i]
            ProL.Set(ProH)
            RerrorReduced3[:,i+cutoff+1] = ProL.vec.FV().NumPy()[:]

    #Clear the variables
    A0H, A1H = None, None
    a0, a1 = None, None
    
########################################################################
#Sort out the error bounds
    if PODErrorBars==True:
        if BigProblem == True:
            MR1 = np.zeros([ndof0,cutoff*2+1],dtype=np.complex64)
            MR2 = np.zeros([ndof0,cutoff*2+1],dtype=np.complex64)
            MR3 = np.zeros([ndof0,cutoff*2+1],dtype=np.complex64)
        else:
            MR1 = np.zeros([ndof0,cutoff*2+1],dtype=complex)
            MR2 = np.zeros([ndof0,cutoff*2+1],dtype=complex)
            MR3 = np.zeros([ndof0,cutoff*2+1],dtype=complex)
        
        u, v = fes0.TnT()
        
        m = BilinearForm(fes0)
        m += SymbolicBFI(InnerProduct(u,v))
        f = LinearForm(fes0)
        m.Assemble()
        c = Preconditioner(m,"local")
        c.Update()
        inverse = CGSolver(m.mat,c.mat,precision=1e-20,maxsteps=500)
        
        ErrorGFU = GridFunction(fes0)
        for i in range(2*cutoff+1):
            #E1
            ProL.vec.data.FV().NumPy()[:] = RerrorReduced1[:,i]
            ProL.vec.data -= m.mat * ErrorGFU.vec
            ErrorGFU.vec.data += inverse * ProL.vec
            MR1[:,i] = ErrorGFU.vec.FV().NumPy()
        
            #E2
            ProL.vec.data.FV().NumPy()[:] = RerrorReduced2[:,i]
            ProL.vec.data -= m.mat * ErrorGFU.vec
            ErrorGFU.vec.data += inverse * ProL.vec
            MR2[:,i] = ErrorGFU.vec.FV().NumPy()
        
            #E3
            ProL.vec.data.FV().NumPy()[:] = RerrorReduced3[:,i]
            ProL.vec.data -= m.mat * ErrorGFU.vec
            ErrorGFU.vec.data += inverse * ProL.vec
            MR3[:,i] = ErrorGFU.vec.FV().NumPy()
        
        
        G_Store = np.zeros([2*cutoff+1,2*cutoff+1,6],dtype = complex)
        G_Store[:,:,0] = np.transpose(np.conjugate(RerrorReduced1))@MR1
        G_Store[:,:,1] = np.transpose(np.conjugate(RerrorReduced2))@MR2
        G_Store[:,:,2] = np.transpose(np.conjugate(RerrorReduced3))@MR3
        G_Store[:,:,3] = np.transpose(np.conjugate(RerrorReduced1))@MR2
        G_Store[:,:,4] = np.transpose(np.conjugate(RerrorReduced1))@MR3
        G_Store[:,:,5] = np.transpose(np.conjugate(RerrorReduced2))@MR3
        
        #Clear the variables
        RerrorReduced1, RerrorReduced2, RerrorReduced3 = None, None, None
        MR1, MR2, MR3 = None, None, None
        fes0, m, c, inverse = None, None, None, None
        
        
        fes3 = HCurl(mesh, order=Order, dirichlet="outer", gradientdomains=dom_nrs_metal)
        ndof3 = fes3.ndof
        Omega = Array[0]
        u,v = fes3.TnT()
        amax = BilinearForm(fes3)
        amax += (mu**(-1))*curl(u)*curl(v)*dx
        amax += (1-inout)*epsi*u*v*dx
        amax += inout*sigma*(alpha**2)*Mu0*Omega*u*v*dx

        m = BilinearForm(fes3)
        m += u*v*dx

        apre = BilinearForm(fes3)
        apre += curl(u)*curl(v)*dx + u*v*dx
        pre = Preconditioner(amax, "bddc")

        with TaskManager():
            amax.Assemble()
            m.Assemble()
            apre.Assemble()

            # build gradient matrix as sparse matrix (and corresponding scalar FESpace)
            gradmat, fesh1 = fes3.CreateGradient()
            gradmattrans = gradmat.CreateTranspose() # transpose sparse matrix
            math1 = gradmattrans @ m.mat @ gradmat   # multiply matrices
            math1[0,0] += 1     # fix the 1-dim kernel
            invh1 = math1.Inverse(inverse="sparsecholesky")
        
            # build the Poisson projector with operator Algebra:
            proj = IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ m.mat
            projpre = proj @ pre.mat
            evals, evecs = solvers.PINVIT(amax.mat, m.mat, pre=projpre, num=1, maxit=50)

        alphaLB = evals[0]
    else:
        alphaLB, G_Store = False, False
    
        #Clear the variables
        fes3, amax, apre, pre, invh1, m = None, None, None, None, None, None
        

######################################################################
#Produce the sweep on the lower dimensional space
    g = np.zeros([cutoff,NumberofFrequencies,3],dtype = complex)
    for k,omega in enumerate(Array):
        g[:,k,0] = np.linalg.solve(HA0H1+HA1H1*omega,HR1*omega)
        g[:,k,1] = np.linalg.solve(HA0H2+HA1H2*omega,HR2*omega)
        g[:,k,2] = np.linalg.solve(HA0H3+HA1H3*omega,HR3*omega)
    #Work out where to send each frequency
    Tensor_CPUs = min(NumberofFrequencies,multiprocessing.cpu_count(),CPUs)
    Core_Distribution = []
    Count_Distribution = []
    for i in range(Tensor_CPUs):
        Core_Distribution.append([])
        Count_Distribution.append([])
    #Distribute frequencies between the cores
    CoreNumber = 0
    for i,Omega in enumerate(Array):
        Core_Distribution[CoreNumber].append(Omega)
        Count_Distribution[CoreNumber].append(i)
        if CoreNumber == Tensor_CPUs-1:
            CoreNumber = 0
        else:
            CoreNumber += 1
    #Distribute the lower dimensional solutions
    Lower_Sols = []
    for i in range(Tensor_CPUs):
        TempArray = np.zeros([cutoff,len(Count_Distribution[i]),3],dtype = complex)
        for j,Sim in enumerate(Count_Distribution[i]):
            TempArray[:,j,:] = g[:,Sim,:]
        Lower_Sols.append(TempArray)
    
    #Cteate the inputs
    Runlist = []
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)
    for i in range(Tensor_CPUs):
        Runlist.append((Core_Distribution[i],mesh,fes,fes2,Lower_Sols[i],u1Truncated,u2Truncated,u3Truncated,Theta0Sol,xivec,alpha,sigma,mu,inout,N0,NumberofFrequencies,counter,PODErrorBars,alphaLB,G_Store))
        
    #Run on the multiple cores
    # Edit James Elgy: changed how pool was generated to 'spawn': see
    # https://britishgeologicalsurvey.github.io/science/python-forking-vs-spawn/
    with multiprocessing.get_context('spawn').Pool(Tensor_CPUs) as pool:
        Outputs = pool.starmap(Theta1_Lower_Sweep, Runlist)

    try:
        pool.terminate()
        print('manually closed pool')
    except:
        print('Pool has already closed.')


    #Unpack the outputs
    if PODErrorBars == True:
        ErrorTensors=np.zeros([NumberofFrequencies,6])
    for i,Output in enumerate(Outputs):
        for j,Num in enumerate(Count_Distribution[i]):
            if PODErrorBars == True:
                TensorArray[Num,:] = Output[0][j]
                EigenValues[Num,:] = Output[1][j]
                ErrorTensors[Num,:] = Output[2][j]
            else:
                TensorArray[Num,:] = Output[0][j]
                EigenValues[Num,:] = Output[1][j]
    
    print(' reduced order systems solved          ')
    print(' frequency sweep complete')
    if PlotPod==True:
        if PODErrorBars==True:
            return TensorArray, EigenValues, N0, PODTensors, PODEigenValues, numelements, ErrorTensors, (ndof, ndof2)
        else:
            return TensorArray, EigenValues, N0, PODTensors, PODEigenValues, numelements, (ndof, ndof2)
    else:
        if PODErrorBars==True:
            return TensorArray, EigenValues, N0, numelements, ErrorTensors, (ndof, ndof2)
        else:
            return TensorArray, EigenValues, N0, numelements, (ndof, ndof2)
