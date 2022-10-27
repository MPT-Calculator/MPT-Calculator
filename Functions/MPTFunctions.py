#This file contains functions used for solving transmission problems and calculating tensors
#Functions -Theta0
#          -Theta1
#          -MPTCalculator
#Importing
import numpy as np
from ngsolve import *

    
#Function definition to solve the Theta0 problem
#Output -The solution to the theta0 problem as a (NGSolve) gridfunction
def Theta0(fes,Order,alpha,mu,inout,e,Tolerance,Maxsteps,epsi,simnumber,Solver, use_longdouble=True):
    #print the progress
    try:
        print(' solving theta0 %d/3' % (simnumber), end='\r')
    except:
        print(' solving the theta0 problem', end='\r')
        
    Theta=GridFunction(fes) 
    Theta.Set((0,0,0), BND)
    
    #Test and trial functions
    u = fes.TrialFunction()
    v = fes.TestFunction()

    #Create the bilinear form (this is for theta^0 tilda)
    f = LinearForm(fes)
    f += SymbolicLFI(inout*(2*(1-mu**(-1)))*InnerProduct(e,curl(v)))
    a = BilinearForm(fes, symmetric=True, condense=True)
    a += SymbolicBFI((mu**(-1))*(curl(u)*curl(v)))
    a += SymbolicBFI(epsi*(u*v))
    if Solver=="bddc":
        c = Preconditioner(a,"bddc")#Apply the bddc preconditioner
    a.Assemble()
    f.Assemble()
    if Solver=="local":
        c = Preconditioner(a,"local")#Apply the local preconditioner
    c.Update()
    
    #Solve the problem
    f.vec.data += a.harmonic_extension_trans * f.vec
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * Theta.vec
    inverse = CGSolver(a.mat, c.mat, precision=Tolerance, maxsteps=Maxsteps, printrates=True)
    Theta.vec.data += inverse * res
    Theta.vec.data += a.inner_solve * f.vec
    Theta.vec.data += a.harmonic_extension * Theta.vec
    
    Theta_Return = np.zeros([fes.ndof], dtype=np.longdouble)
    Theta_Return[:] = Theta.vec.FV().NumPy()
    return Theta_Return



#Function definition to solve the Theta1 problem
#Output -The solution to the theta1 problem as a (NGSolve) gridfunction
def Theta1(fes,fes2,Theta0Sol,xi,Order,alpha,nu,sigma,mu,inout,Tolerance,Maxsteps,epsi,Omega,simnumber,outof,Solver):
    #print the counter
    try:#This is used for the simulations run in parallel
        simnumber.value+=1
        print(' solving theta1 %d/%d    ' % (floor((simnumber.value)/3),outof), end='\r')
    except:
        try:#This is for the simulations run consecutively and the single frequency case
            print(' solving theta1 %d/%d    ' % (simnumber,outof), end='\r')
        except:# This is for the single frequency case with 3 CPUs
            print(' solving the theta1 problem  ', end='\r')
        
    Theta0=GridFunction(fes)
    Theta0.vec.FV().NumPy()[:]=Theta0Sol
    Theta=GridFunction(fes2)
    Theta.Set((0,0,0), BND)

    #Test and trial functions
    u = fes2.TrialFunction()
    v = fes2.TestFunction()

    #Create the bilinear form (this is for the complex conjugate of theta^1)
    f = LinearForm(fes2)
    f += SymbolicLFI(inout * (-1j) * nu*sigma * InnerProduct(Theta0,v))
    f += SymbolicLFI(inout * (-1j) * nu*sigma * InnerProduct(xi,v))
    a = BilinearForm(fes2, symmetric=True, condense=True)
    a += SymbolicBFI((mu**(-1)) * InnerProduct(curl(u),curl(v)))
    a += SymbolicBFI((1j) * inout * nu*sigma * InnerProduct(u,v))
    a += SymbolicBFI((1j) * (1-inout) * epsi * InnerProduct(u,v))
    if Solver=="bddc":
        c = Preconditioner(a,"bddc")#Apply the bddc preconditioner
    a.Assemble()
    f.Assemble()
    if Solver=="local":
        c = Preconditioner(a,"local")#Apply the local preconditioner
    c.Update()

    #Solve
    f.vec.data += a.harmonic_extension_trans * f.vec
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * Theta.vec
    inverse = CGSolver(a.mat, c.mat, precision=Tolerance, maxsteps=Maxsteps)
    Theta.vec.data += inverse * res
    Theta.vec.data += a.inner_solve * f.vec
    Theta.vec.data += a.harmonic_extension * Theta.vec
    
    Theta_Return = np.zeros([fes2.ndof],dtype=np.clongdouble)
    Theta_Return[:] = Theta.vec.FV().NumPy()
    return Theta_Return



def Theta1_Sweep(Array,mesh,fes,fes2,Theta0Sols,xivec,alpha,sigma,mu,inout,Tolerance,Maxsteps,epsi,Solver,N0,TotalNOF,Vectors,Tensors,Multi,BP):
    print(' solving theta1')
    #Setup variables
    Mu0 = 4*np.pi*10**(-7)
    nu_no_omega = Mu0*(alpha**2)
    NOF = len(Array)

    #Setup where to store tensors
    if Tensors == True:
        R=np.zeros([3,3])
        I=np.zeros([3,3])
        TensorArray=np.zeros([NOF,9], dtype=complex)
        EigenValues = np.zeros([NOF,3], dtype=complex)

    #Setup where to save the solution vectors
    if Vectors == True:
        ndof = fes2.ndof
        if BP==True:
            Theta1Sols = np.zeros([ndof,NOF,3],dtype=np.complex64)
        else:
            Theta1Sols = np.zeros([ndof,NOF,3],dtype=complex)

    Theta0i = GridFunction(fes)
    Theta0j = GridFunction(fes)
    Theta1i = GridFunction(fes2)
    Theta1j = GridFunction(fes2)
    Theta1 = GridFunction(fes2)
    Theta2 = GridFunction(fes2)
    Theta3 = GridFunction(fes2)

    #Test and trial functions
    u,v = fes2.TnT()

    #Setup righthand sides
    Theta0i.vec.FV().NumPy()[:] = Theta0Sols[:,0]
    f1 = LinearForm(fes2)
    f1 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta0i,v))
    f1 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[0],v))
    f1.Assemble()

    Theta0i.vec.FV().NumPy()[:] = Theta0Sols[:,1]
    f2 = LinearForm(fes2)
    f2 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta0i,v))
    f2 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[1],v))
    f2.Assemble()

    Theta0i.vec.FV().NumPy()[:] = Theta0Sols[:,2]
    f3 = LinearForm(fes2)
    f3 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta0i,v))
    f3 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[2],v))
    f3.Assemble()

    #Set up a vector for the residual and solving
    res = f1.vec.CreateVector()
    ftemp = f1.vec.CreateVector()



    for k,Omega in enumerate(Array):
        if Multi == False:
            print(' solving theta1 %d/%d    ' % (k+1,NOF), end='\r')
        else:
            try:
                Multi.value += 1
                print(' solving theta1 %d/%d    ' % (Multi.value,TotalNOF), end='\r')
            except:
                print(' solving theta1', end='\r')

        #Create the bilinear form
        a = BilinearForm(fes2, symmetric=True, condense=True)
        a += SymbolicBFI((mu**(-1)) * InnerProduct(curl(u),curl(v)))
        a += SymbolicBFI((1j) * inout * nu_no_omega * Omega * sigma * InnerProduct(u,v))
        a += SymbolicBFI((1j) * (1-inout) * epsi * InnerProduct(u,v))
        if Solver == "bddc":
            c = Preconditioner(a,"bddc")#Apply the bddc preconditioner
        a.Assemble()
        if Solver == "local":
            c = Preconditioner(a,"local")#Apply the local preconditioner
        c.Update()

        #Calculate the inverse operator
        inverse = CGSolver(a.mat, c.mat, precision=Tolerance, maxsteps=Maxsteps)
    
        #Solve in each direction
        
        Theta1.Set((0,0,0), BND)
        Theta2.Set((0,0,0), BND)
        Theta3.Set((0,0,0), BND)
        
        #e1
        res.data.FV().NumPy()[:] = f1.vec.FV().NumPy() * Omega
        res.data += a.harmonic_extension_trans * res.data
        ftemp.data = res.data
        res.data -= a.mat * Theta1.vec
        Theta1.vec.data += inverse * res
        Theta1.vec.data += a.inner_solve * ftemp.data
        Theta1.vec.data += a.harmonic_extension * Theta1.vec
        
        #e2
        res.data.FV().NumPy()[:] = f2.vec.FV().NumPy() * Omega
        res.data += a.harmonic_extension_trans * res.data
        ftemp.data = res.data
        res.data -= a.mat * Theta2.vec
        Theta2.vec.data += inverse * res
        Theta2.vec.data += a.inner_solve * ftemp.data
        Theta2.vec.data += a.harmonic_extension * Theta2.vec
        
        #e3
        res.data.FV().NumPy()[:] = f3.vec.FV().NumPy() * Omega
        res.data += a.harmonic_extension_trans * res.data
        ftemp.data = res.data
        res.data -= a.mat * Theta3.vec
        Theta3.vec.data += inverse * res
        Theta3.vec.data += a.inner_solve * ftemp.data
        Theta3.vec.data += a.harmonic_extension * Theta3.vec
    
        if Vectors == True:
            Theta1Sols[:,k,0] = Theta1.vec.FV().NumPy()
            Theta1Sols[:,k,1] = Theta2.vec.FV().NumPy()
            Theta1Sols[:,k,2] = Theta3.vec.FV().NumPy()

        if Tensors == True:
            #Calculate upper triangle of tensor
            R=np.zeros([3,3])
            I=np.zeros([3,3])
            for i in range(3):
                Theta0i.vec.FV().NumPy()[:] = Theta0Sols[:,i]
                xii = xivec[i]
                if i==0:
                    Theta1i.vec.data=Theta1.vec.data
                if i==1:
                    Theta1i.vec.data=Theta2.vec.data
                if i==2:
                    Theta1i.vec.data=Theta3.vec.data
                for j in range(i+1):
                    Theta0j.vec.FV().NumPy()[:] = Theta0Sols[:,j]
                    xij = xivec[j]
                    if j==0:
                        Theta1j.vec.data=Theta1.vec.data
                    if j==1:
                        Theta1j.vec.data=Theta2.vec.data
                    if j==2:
                        Theta1j.vec.data=Theta3.vec.data
        
                    #Real and Imaginary parts
                    R[i,j]=-(((alpha**3)/4)*Integrate((mu**(-1))*(curl(Theta1j)*Conj(curl(Theta1i))),mesh)).real
                    I[i,j]=((alpha**3)/4)*Integrate(inout*nu_no_omega*Omega*sigma*((Theta1j+Theta0j+xij)*(Conj(Theta1i)+Theta0i+xii)),mesh).real

            #Mirror tensor
            R+=np.transpose(R-np.diag(np.diag(R)))
            I+=np.transpose(I-np.diag(np.diag(I)))

            #Save in arrays
            TensorArray[k,:] = (N0+R+1j*I).flatten()
            EigenValues[k,:] = np.sort(np.linalg.eigvals(N0+R))+1j*np.sort(np.linalg.eigvals(I))

    if Tensors == True and Vectors == True:
        return TensorArray, EigenValues, Theta1Sols
    elif Tensors == True:
        return TensorArray, EigenValues
    else:
        return Theta1Sols



def Theta1_Lower_Sweep(Array,mesh,fes,fes2,Sols,u1Truncated,u2Truncated,u3Truncated,Theta0Sols,xivec,alpha,sigma,mu,inout,N0,TotalNOF,counter,PODErrorBars,alphaLB,G_Store):
    
    #Setup variables
    Mu0 = 4*np.pi*10**(-7)
    nu_no_omega = Mu0*(alpha**2)
    NOF = len(Array)
    cutoff = len(Sols[:,0,0])
    Theta_0i=GridFunction(fes)
    Theta_0j=GridFunction(fes)
    Theta_1i=GridFunction(fes2)
    Theta_1j=GridFunction(fes2)
    TensorArray = np.zeros([NOF,9],dtype=complex)
    EigenValues = np.zeros([NOF,3],dtype=complex)
    
    if PODErrorBars == True:
        rom1 = np.zeros([1+2*cutoff,1],dtype=complex)
        rom2 = np.zeros([1+2*cutoff,1],dtype=complex)
        rom3 = np.zeros([1+2*cutoff,1],dtype=complex)
        TensorErrors=np.zeros([NOF,3])
        ErrorTensors=np.zeros([NOF,6])
        G1 = G_Store[:,:,0]
        G2 = G_Store[:,:,1]
        G3 = G_Store[:,:,2]
        G12 = G_Store[:,:,3]
        G13 = G_Store[:,:,4]
        G23 = G_Store[:,:,5]
    
    for k,omega in enumerate(Array):

        #This part is for obtaining the solutions in the lower dimensional space
        try:
            counter.value+=1
            print(' solving reduced order system %d/%d    ' % (counter.value,TotalNOF), end='\r')
        except:
            print(' solving reduced order system', end='\r')

        #This part projects the problem to the higher dimensional space
        W1=np.dot(u1Truncated,Sols[:,k,0]).flatten()
        W2=np.dot(u2Truncated,Sols[:,k,1]).flatten()
        W3=np.dot(u3Truncated,Sols[:,k,2]).flatten()
        
        #Calculate the tensors
        nu = omega*Mu0*(alpha**2)
        R=np.zeros([3,3])
        I=np.zeros([3,3])
        
        for i in range(3):
            Theta_0i.vec.FV().NumPy()[:] = Theta0Sols[:,i]
            xii = xivec[i]
            if i==0:
                Theta_1i.vec.FV().NumPy()[:]=W1
            if i==1:
                Theta_1i.vec.FV().NumPy()[:]=W2
            if i==2:
                Theta_1i.vec.FV().NumPy()[:]=W3
            for j in range(i+1):
                Theta_0j.vec.FV().NumPy()[:]=Theta0Sols[:,j]
                xij=xivec[j]
                if j==0:
                    Theta_1j.vec.FV().NumPy()[:]=W1
                if j==1:
                    Theta_1j.vec.FV().NumPy()[:]=W2
                if j==2:
                    Theta_1j.vec.FV().NumPy()[:]=W3
                
                #Real and Imaginary parts
                R[i,j]=-(((alpha**3)/4)*Integrate((mu**(-1))*(curl(Theta_1j)*Conj(curl(Theta_1i))),mesh)).real
                I[i,j]=((alpha**3)/4)*Integrate(inout*nu*sigma*((Theta_1j+Theta_0j+xij)*(Conj(Theta_1i)+Theta_0i+xii)),mesh).real
        
        R+=np.transpose(R-np.diag(np.diag(R))).real
        I+=np.transpose(I-np.diag(np.diag(I))).real

        #Save in arrays
        TensorArray[k,:] = (N0+R+1j*I).flatten()
        EigenValues[k,:] = np.sort(np.linalg.eigvals(N0+R))+1j*np.sort(np.linalg.eigvals(I))
        
        
        if PODErrorBars==True:
            rom1[0,0] = omega
            rom2[0,0] = omega
            rom3[0,0] = omega

            rom1[1:1+cutoff,0] = -Sols[:,k,0].flatten()
            rom2[1:1+cutoff,0] = -Sols[:,k,1].flatten()
            rom3[1:1+cutoff,0] = -Sols[:,k,2].flatten()

            rom1[1+cutoff:,0] = -(Sols[:,k,0]*omega).flatten()
            rom2[1+cutoff:,0] = -(Sols[:,k,1]*omega).flatten()
            rom3[1+cutoff:,0] = -(Sols[:,k,2]*omega).flatten()
            
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
    
    
    if PODErrorBars == True:
        return TensorArray, EigenValues, ErrorTensors
    else:
        return TensorArray, EigenValues




#Function definition to calculate MPTs from solution vectors
#Outputs -R as a numpy array
#        -I as a numpy array (This contains real values not imaginary ones)
def MPTCalculator(mesh,fes,fes2,Theta1E1Sol,Theta1E2Sol,Theta1E3Sol,Theta0Sol,xivec,alpha,mu,sigma,inout,nu,tennumber,outof):
    #Print the progress of the sweep
    try:#This is used for the simulations run in parallel
        tennumber.value+=1
        print(' calculating tensor %d/%d    ' % (tennumber.value,outof), end='\r')
    except:#This is for the POD run consecutively
        try:
            print(' calculating tensor %d/%d    ' % (tennumber,outof), end='\r')
        except:#This is for the full sweep run consecutively (no print)
            pass
    
    R=np.zeros([3,3])
    I=np.zeros([3,3])
    Theta0i=GridFunction(fes)
    Theta0j=GridFunction(fes)
    Theta1i=GridFunction(fes2)
    Theta1j=GridFunction(fes2)
    for i in range(3):
        Theta0i.vec.FV().NumPy()[:]=Theta0Sol[:,i]
        xii=xivec[i]
        if i==0:
            Theta1i.vec.FV().NumPy()[:]=Theta1E1Sol
        if i==1:
            Theta1i.vec.FV().NumPy()[:]=Theta1E2Sol
        if i==2:
            Theta1i.vec.FV().NumPy()[:]=Theta1E3Sol
        for j in range(i+1):
            Theta0j.vec.FV().NumPy()[:]=Theta0Sol[:,j]
            xij=xivec[j]
            if j==0:
                Theta1j.vec.FV().NumPy()[:]=Theta1E1Sol
            if j==1:
                Theta1j.vec.FV().NumPy()[:]=Theta1E2Sol
            if j==2:
                Theta1j.vec.FV().NumPy()[:]=Theta1E3Sol

            #Real and Imaginary parts
            R[i,j]=-(((alpha**3)/4)*Integrate((mu**(-1))*(curl(Theta1j)*Conj(curl(Theta1i))),mesh)).real
            I[i,j]=((alpha**3)/4)*Integrate(inout*nu*sigma*((Theta1j+Theta0j+xij)*(Conj(Theta1i)+Theta0i+xii)),mesh).real
    R+=np.transpose(R-np.diag(np.diag(R))).real
    I+=np.transpose(I-np.diag(np.diag(I))).real
    return R, I























