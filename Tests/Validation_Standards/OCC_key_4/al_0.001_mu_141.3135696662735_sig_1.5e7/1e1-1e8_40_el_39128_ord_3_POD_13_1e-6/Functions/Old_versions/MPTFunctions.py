#This file contains functions used for solving transmission problems and calculating tensors
#Functions -Theta0
#          -Theta1
#          -MPTCalculator
#Importing
import numpy as np
from ngsolve import *
import scipy.sparse as sp
import gc

    
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

    del f, a, c, res, u, v, inverse, Theta
    gc.collect()

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

    del f, a, c, res, u, v, Theta
    gc.collect()

    return Theta_Return



def Theta1_Sweep(Array,mesh,fes,fes2,Theta0Sols,xivec,alpha,sigma,mu,inout,Tolerance,Maxsteps,epsi,Solver,N0,TotalNOF,Vectors,Tensors,Multi,BP, Order):
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
                    R[i,j]=-(((alpha**3)/4)*Integrate((mu**(-1))*(curl(Theta1j)*Conj(curl(Theta1i))),mesh, order=2*(Order+1))).real
                    I[i,j]=((alpha**3)/4)*Integrate(inout*nu_no_omega*Omega*sigma*((Theta1j+Theta0j+xij)*(Conj(Theta1i)+Theta0i+xii)),mesh, order=2*(Order+1)).real

            #Mirror tensor
            R+=np.transpose(R-np.diag(np.diag(R)))
            I+=np.transpose(I-np.diag(np.diag(I)))

            #Save in arrays
            TensorArray[k,:] = (N0+R+1j*I).flatten()
            EigenValues[k,:] = np.sort(np.linalg.eigvals(N0+R))+1j*np.sort(np.linalg.eigvals(I))

    del f1, f2, f3, ftemp
    del res, a, c
    del Theta0i, Theta1i, Theta0j, Theta1j
    gc.collect()


    if Tensors == True and Vectors == True:
        return TensorArray, EigenValues, Theta1Sols
    elif Tensors == True:
        return TensorArray, EigenValues
    else:
        return Theta1Sols



def Theta1_Lower_Sweep(Array, mesh, fes, fes2, Sols, u1Truncated, u2Truncated, u3Truncated, Theta0Sols, xivec, alpha,
                       sigma, mu, inout, N0, TotalNOF, counter, PODErrorBars, alphaLB, G_Store, Order, AdditionalInt,
                       use_integral):
    # Setup variables
    Mu0 = 4 * np.pi * 10 ** (-7)
    nu_no_omega = Mu0 * (alpha ** 2)
    NOF = len(Array)
    cutoff = len(Sols[:, 0, 0])
    Theta_0i = GridFunction(fes)
    Theta_0j = GridFunction(fes)
    Theta_1i = GridFunction(fes2)
    Theta_1j = GridFunction(fes2)
    TensorArray = np.zeros([NOF, 9], dtype=complex)
    EigenValues = np.zeros([NOF, 3], dtype=complex)

    if PODErrorBars == True:
        rom1 = np.zeros([1 + 2 * cutoff, 1], dtype=complex)
        rom2 = np.zeros([1 + 2 * cutoff, 1], dtype=complex)
        rom3 = np.zeros([1 + 2 * cutoff, 1], dtype=complex)
        TensorErrors = np.zeros([NOF, 3])
        ErrorTensors = np.zeros([NOF, 6])
        G1 = G_Store[:, :, 0]
        G2 = G_Store[:, :, 1]
        G3 = G_Store[:, :, 2]
        G12 = G_Store[:, :, 3]
        G13 = G_Store[:, :, 4]
        G23 = G_Store[:, :, 5]

    # Edit James Elgy 2022 - Calculate R and I via explicit matrix multiplication rather than integrals.
    # use_integral = False
    # Faster numerical computation is acheivable via the Theta1_Lower_Sweep_Mat_Method function, but this code has been
    # left in as a well tested fallback and for the computation of the error cert
    if use_integral is False:
        u, v = fes2.TnT()
        K = BilinearForm(fes2, symmetric=True)
        K += SymbolicBFI(inout * mu ** (-1) * curl(u) * curl(v), bonus_intorder=AdditionalInt)
        K += SymbolicBFI((1 - inout) * curl(u) * curl(v), bonus_intorder=AdditionalInt)
        K.Assemble()

        A = BilinearForm(fes2, symmetric=True)
        A += SymbolicBFI(sigma * inout * (v * u), bonus_intorder=AdditionalInt)
        A.Assemble()
        rows, cols, vals = A.mat.COO()
        A_mat = sp.csr_matrix((vals, (rows, cols)))

        E = np.zeros((3, fes2.ndof), dtype=complex)
        G = np.zeros((3, 3))

        for i in range(3):

            E_lf = LinearForm(fes2)
            E_lf += SymbolicLFI(sigma * inout * xivec[i] * v, bonus_intorder=AdditionalInt)
            E_lf.Assemble()
            E[i, :] = E_lf.vec.FV().NumPy()[:]

            for j in range(3):
                G[i, j] = Integrate(sigma * inout * xivec[i] * xivec[j], mesh, order=2 * (Order + 1))

            H = E.transpose()

        rows, cols, vals = K.mat.COO()
        Q = sp.csr_matrix((vals, (rows, cols)))
        del E_lf
        del A
        del K

        # For faster computation of tensor coefficients, we multiply with Ui before the loop.
        Q11 = np.conj(np.transpose(u1Truncated)) @ Q @ u1Truncated
        Q22 = np.conj(np.transpose(u2Truncated)) @ Q @ u2Truncated
        Q33 = np.conj(np.transpose(u3Truncated)) @ Q @ u3Truncated
        Q21 = np.conj(np.transpose(u2Truncated)) @ Q @ u1Truncated
        Q31 = np.conj(np.transpose(u3Truncated)) @ Q @ u1Truncated
        Q32 = np.conj(np.transpose(u3Truncated)) @ Q @ u2Truncated

        # Similarly for the imaginary part, we multiply with the theta0 sols beforehand.
        A_mat_t0_1 = (A_mat) @ Theta0Sols[:, 0]
        A_mat_t0_2 = (A_mat) @ Theta0Sols[:, 1]
        A_mat_t0_3 = (A_mat) @ Theta0Sols[:, 2]

        c1_11 = (np.transpose(Theta0Sols[:, 0])) @ A_mat_t0_1
        c1_22 = (np.transpose(Theta0Sols[:, 1])) @ A_mat_t0_2
        c1_33 = (np.transpose(Theta0Sols[:, 2])) @ A_mat_t0_3
        c1_21 = (np.transpose(Theta0Sols[:, 1])) @ A_mat_t0_1
        c1_31 = (np.transpose(Theta0Sols[:, 2])) @ A_mat_t0_1
        c1_32 = (np.transpose(Theta0Sols[:, 2])) @ A_mat_t0_2

        c5_11 = E[0, :] @ Theta0Sols[:, 0]
        c5_22 = E[1, :] @ Theta0Sols[:, 1]
        c5_33 = E[2, :] @ Theta0Sols[:, 2]
        c5_21 = E[1, :] @ Theta0Sols[:, 0]
        c5_31 = E[2, :] @ Theta0Sols[:, 0]
        c5_32 = E[2, :] @ Theta0Sols[:, 1]

        T11 = np.conj(np.transpose(u1Truncated)) @ A_mat @ u1Truncated
        T22 = np.conj(np.transpose(u2Truncated)) @ A_mat @ u2Truncated
        T33 = np.conj(np.transpose(u3Truncated)) @ A_mat @ u3Truncated
        T21 = np.conj(np.transpose(u2Truncated)) @ A_mat @ u1Truncated
        T31 = np.conj(np.transpose(u3Truncated)) @ A_mat @ u1Truncated
        T32 = np.conj(np.transpose(u3Truncated)) @ A_mat @ u2Truncated

    for k, omega in enumerate(Array):

        # This part is for obtaining the solutions in the lower dimensional space
        try:
            counter.value += 1
            print(' solving reduced order system %d/%d    ' % (counter.value, TotalNOF), end='\r')
        except:
            print(' solving reduced order system', end='\r')

        # This part projects the problem to the higher dimensional space
        W1 = np.dot(u1Truncated, Sols[:, k, 0]).flatten()
        W2 = np.dot(u2Truncated, Sols[:, k, 1]).flatten()
        W3 = np.dot(u3Truncated, Sols[:, k, 2]).flatten()

        # Calculate the tensors
        nu = omega * Mu0 * (alpha ** 2)
        R = np.zeros([3, 3])
        I = np.zeros([3, 3])

        if use_integral is True:
            for i in range(3):
                Theta_0i.vec.FV().NumPy()[:] = Theta0Sols[:, i]
                xii = xivec[i]
                if i == 0:
                    Theta_1i.vec.FV().NumPy()[:] = W1
                if i == 1:
                    Theta_1i.vec.FV().NumPy()[:] = W2
                if i == 2:
                    Theta_1i.vec.FV().NumPy()[:] = W3
                for j in range(i + 1):
                    Theta_0j.vec.FV().NumPy()[:] = Theta0Sols[:, j]
                    xij = xivec[j]
                    if j == 0:
                        Theta_1j.vec.FV().NumPy()[:] = W1
                    if j == 1:
                        Theta_1j.vec.FV().NumPy()[:] = W2
                    if j == 2:
                        Theta_1j.vec.FV().NumPy()[:] = W3

                    # Real and Imaginary parts
                    R[i, j] = -(((alpha ** 3) / 4) * Integrate((mu ** (-1)) * (curl(Theta_1j) * Conj(curl(Theta_1i))),
                                                               mesh, order=2 * (Order + 1))).real
                    I[i, j] = ((alpha ** 3) / 4) * Integrate(
                        inout * nu * sigma * ((Theta_1j + Theta_0j + xij) * (Conj(Theta_1i) + Theta_0i + xii)), mesh,
                        order=2 * (Order + 1)).real

        # Use matrix method.
        else:
            for i in range(3):
                for i in range(3):
                    t0i = Theta0Sols[:, i] + 1j * np.zeros(Theta0Sols[:, i].shape)
                    if i == 0:
                        gi = np.squeeze(Sols[:, k, 0])
                        wi = W1
                    elif i == 1:
                        gi = np.squeeze(Sols[:, k, 1])
                        wi = W2
                    elif i == 2:
                        gi = np.squeeze(Sols[:, k, 2])
                        wi = W3

                    for j in range(i + 1):
                        t0j = Theta0Sols[:, j] + 1j * np.zeros(Theta0Sols[:, j].shape)
                        if j == 0:
                            gj = np.squeeze(Sols[:, k, 0])
                            wj = W1
                        elif j == 1:
                            gj = np.squeeze(Sols[:, k, 1])
                            wj = W2
                        elif j == 2:
                            gj = np.squeeze(Sols[:, k, 2])
                            wj = W3

                        if i == 0 and j == 0:
                            Q = Q11
                            T = T11
                            c1 = c1_11
                            A_mat_t0 = A_mat_t0_1
                            c5 = c5_11
                        elif i == 1 and j == 1:
                            Q = Q22
                            T = T22
                            c1 = c1_22
                            A_mat_t0 = A_mat_t0_2
                            c5 = c5_22
                        elif i == 2 and j == 2:
                            Q = Q33
                            c1 = c1_33
                            T = T33
                            A_mat_t0 = A_mat_t0_3
                            c5 = c5_33

                        elif i == 1 and j == 0:
                            Q = Q21
                            T = T21
                            c1 = c1_21
                            A_mat_t0 = A_mat_t0_1
                            c5 = c5_21
                        elif i == 2 and j == 0:
                            Q = Q31
                            T = T31
                            c1 = c1_31
                            A_mat_t0 = A_mat_t0_1
                            c5 = c5_31
                        elif i == 2 and j == 1:
                            Q = Q32
                            T = T32
                            c1 = c1_32
                            A_mat_t0 = A_mat_t0_2
                            c5 = c5_32
                        # # Looping through non-zero entries in sparse K matrix.
                        # A = 0
                        # Z = np.zeros(fes2.ndof, dtype=complex)
                        # for row_ele, col_ele, val_ele in zip(rows, cols, vals):
                        #     # Post multiplication Z = K u
                        #     Z[row_ele] += val_ele * np.conj(t1j[col_ele])
                        # for row_ele in rows:
                        #     # Pre multiplication A = u^T Z
                        #     A += (t1i[row_ele] * (Z[row_ele]))

                        A = np.conj(gi[None, :]) @ Q @ (gj)[:, None]
                        R[i, j] = (A * (-alpha ** 3) / 4).real

                        # # c1 = (t0i)[None, :] @ (A_mat * omega) @ (t0j)[:, None]
                        # # c2 = (t1i)[None, :] @ (A_mat * omega) @ (t0j)[:, None]
                        c2 = wi[None, :] @ A_mat_t0
                        # # c3 = (t0i)[None, :] @ (A_mat * omega) @ np.conj(t1j)[:, None]
                        c3 = (t0i)[None, :] @ A_mat @ np.conj(wj)[:, None]
                        # # c4 = (t1i)[None, :] @ (A_mat * omega) @ np.conj(t1j)[:, None]
                        c4 = (wi)[None, :] @ A_mat @ np.conj(wj)[:, None]
                        # # c5 = (E[i, :] * omega) @ t0j[:, None]
                        # c5 = (E[i, :]) @ t0j[:, None]
                        # # c6 = (E[i, :] * omega) @ np.conj(t1j)[:, None]
                        c6 = (E[i, :]) @ np.conj(wj)[:, None]
                        # # c7 = G[i, j] * omega
                        # c7 = G[i, j]
                        # # c8 = (t0i[None, :] @ ((H[:, j] * omega)))
                        c8 = t0i[None, :] @ H[:, j]
                        # # c9 = (t1i[None, :] @ ((H[:, j] * omega)))
                        c9 = wi[None, :] @ H[:, j]
                        #
                        # total = omega * (
                        #             c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9)  # c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9
                        # I[i, j] = ((alpha ** 3) / 4) * complex(total).real

                        # c_sum = np.real(np.conj(gj) @ T @ gi) + 2 * np.real(wi @ A_mat_t0) + 2 * np.real(E[i, :] @ (t0j + np.conj(wj)))

                        c_sum = np.real(c2 + c3 + c4 + c5 + c6 + c8 + c9)

                        I[i, j] = np.real((alpha ** 3 / 4) * omega * Mu0 * alpha ** 2 * (c1 + G[i, j] + c_sum))

        R += np.transpose(R - np.diag(np.diag(R))).real
        I += np.transpose(I - np.diag(np.diag(I))).real

        # Save in arrays
        TensorArray[k, :] = (N0 + R + 1j * I).flatten()
        EigenValues[k, :] = np.sort(np.linalg.eigvals(N0 + R)) + 1j * np.sort(np.linalg.eigvals(I))

        if PODErrorBars == True:
            rom1[0, 0] = omega
            rom2[0, 0] = omega
            rom3[0, 0] = omega

            rom1[1:1 + cutoff, 0] = -Sols[:, k, 0].flatten()
            rom2[1:1 + cutoff, 0] = -Sols[:, k, 1].flatten()
            rom3[1:1 + cutoff, 0] = -Sols[:, k, 2].flatten()

            rom1[1 + cutoff:, 0] = -(Sols[:, k, 0] * omega).flatten()
            rom2[1 + cutoff:, 0] = -(Sols[:, k, 1] * omega).flatten()
            rom3[1 + cutoff:, 0] = -(Sols[:, k, 2] * omega).flatten()

            error1 = np.conjugate(np.transpose(rom1)) @ G1 @ rom1
            error2 = np.conjugate(np.transpose(rom2)) @ G2 @ rom2
            error3 = np.conjugate(np.transpose(rom3)) @ G3 @ rom3
            error12 = np.conjugate(np.transpose(rom1)) @ G12 @ rom2
            error13 = np.conjugate(np.transpose(rom1)) @ G13 @ rom3
            error23 = np.conjugate(np.transpose(rom2)) @ G23 @ rom3

            error1 = abs(error1) ** (1 / 2)
            error2 = abs(error2) ** (1 / 2)
            error3 = abs(error3) ** (1 / 2)
            error12 = error12.real
            error13 = error13.real
            error23 = error23.real

            Errors = [error1, error2, error3, error12, error13, error23]

            for j in range(6):
                if j < 3:
                    ErrorTensors[k, j] = ((alpha ** 3) / 4) * (Errors[j] ** 2) / alphaLB
                else:
                    ErrorTensors[k, j] = -2 * Errors[j]
                    if j == 3:
                        ErrorTensors[k, j] += (Errors[0] ** 2) + (Errors[1] ** 2)
                        ErrorTensors[k, j] = ((alpha ** 3) / (8 * alphaLB)) * (
                                    (Errors[0] ** 2) + (Errors[1] ** 2) + ErrorTensors[k, j])
                    if j == 4:
                        ErrorTensors[k, j] += (Errors[0] ** 2) + (Errors[2] ** 2)
                        ErrorTensors[k, j] = ((alpha ** 3) / (8 * alphaLB)) * (
                                    (Errors[0] ** 2) + (Errors[1] ** 2) + ErrorTensors[k, j])
                    if j == 5:
                        ErrorTensors[k, j] += (Errors[1] ** 2) + (Errors[2] ** 2)
                        ErrorTensors[k, j] = ((alpha ** 3) / (8 * alphaLB)) * (
                                    (Errors[0] ** 2) + (Errors[1] ** 2) + ErrorTensors[k, j])

    if use_integral is False:
        del Q, Q11, Q22, Q33, Q21, Q31, Q32
        del T, T11, T22, T33, T21, T31, T32
        del c1_11, c1_22, c1_33, c1_21, c1_31, c1_32
        del c5_11, c5_22, c5_33, c5_21, c5_31, c5_32
    del Theta_0i, Theta_1i, Theta_0j, Theta_1j
    gc.collect()

    if PODErrorBars == True:
        return TensorArray, EigenValues, ErrorTensors
    else:
        return TensorArray, EigenValues


def Theta1_Lower_Sweep_Mat_Method(Array, Q_array, c1_array, c5_array, c7, c8_array, At0_array, At0U_array, T_array, EU_array, Sols, G_Store, cutoff, NOF, alpha, calc_errortensors):

    if calc_errortensors is True:
        rom1 = np.zeros([1 + 2 * cutoff, 1], dtype=complex)
        rom2 = np.zeros([1 + 2 * cutoff, 1], dtype=complex)
        rom3 = np.zeros([1 + 2 * cutoff, 1], dtype=complex)
        TensorErrors = np.zeros([NOF, 3])
        ErrorTensors = np.zeros([NOF, 6])
        G1 = G_Store[:, :, 0]
        G2 = G_Store[:, :, 1]
        G3 = G_Store[:, :, 2]
        G12 = G_Store[:, :, 3]
        G13 = G_Store[:, :, 4]
        G23 = G_Store[:, :, 5]

    TensorArray_no_N0 = np.zeros((len(Array), 9), dtype=complex)
    for k, omega in enumerate(Array):
        nu = omega * 4*np.pi*1e-7 * (alpha ** 2)
        R = np.zeros([3, 3])
        I = np.zeros([3, 3])

        for i in range(3):
            if i == 0:
                gi = np.squeeze(Sols[:, k, 0])
            elif i == 1:
                gi = np.squeeze(Sols[:, k, 1])
            elif i == 2:
                gi = np.squeeze(Sols[:, k, 2])

            for j in range(i + 1):
                if j == 0:
                    gj = np.squeeze(Sols[:, k, 0])
                elif j == 1:
                    gj = np.squeeze(Sols[:, k, 1])
                elif j == 2:
                    gj = np.squeeze(Sols[:, k, 2])

                if i == j:
                    Q = Q_array[i]
                    T = T_array[i]
                    c1 = c1_array[i]
                    c8 = c8_array[i]
                    A_mat_t0 = At0_array[i]
                    At0U = At0U_array[i]
                    c5 = c5_array[i]
                    EU = EU_array[i]
                elif i == 1 and j == 0:
                    Q = Q_array[3]
                    T = T_array[3]
                    c1 = c1_array[3]
                    At0U = At0U_array[3]
                    c8 = c8_array[3]
                    A_mat_t0 = At0_array[0]
                    c5 = c5_array[3]
                    EU = EU_array[3]
                elif i == 2 and j == 0:
                    Q = Q_array[4]
                    T = T_array[4]
                    At0U = At0U_array[4]
                    c1 = c1_array[4]
                    c8 = c8_array[4]
                    A_mat_t0 = At0_array[0]
                    c5 = c5_array[4]
                    EU = EU_array[4]
                elif i == 2 and j == 1:
                    Q = Q_array[5]
                    T = T_array[5]
                    At0U = At0U_array[5]
                    c1 = c1_array[5]
                    c8 = c8_array[5]
                    A_mat_t0 = At0_array[1]
                    c5 = c5_array[5]
                    EU = EU_array[5]

                # Calc Real Part:
                A = np.conj(gi[None, :]) @ Q @ (gj)[:, None]
                R[i, j] = (A * (-alpha ** 3) / 4).real

                # Calc Imag Part:
                p1 = np.real(np.conj(gi) @ T @ gj)
                p2 = np.real(2 * np.conj(gj.transpose()) @  At0U)
                p3 = np.real(c8 + c5)
                p4 = np.real(2 * EU @ np.conj(gj))
                # p4 += np.real(EU.transpose() @ np.conj(gi.transpose()))

                I[i,j] = np.real((alpha ** 3 / 4) * omega * 4*np.pi*1e-7 * alpha ** 2 * (c1 + c7[i, j] + p1 + p2 + p3 + p4))

        R += np.transpose(R - np.diag(np.diag(R))).real
        I += np.transpose(I - np.diag(np.diag(I))).real

        # Save in arrays
        TensorArray_no_N0[k,:] = (R + 1j * I).flatten()

    return TensorArray_no_N0, 0
#Function definition to calculate MPTs from solution vectors
#Outputs -R as a numpy array
#        -I as a numpy array (This contains real values not imaginary ones)
def MPTCalculator(mesh,fes,fes2,Theta1E1Sol,Theta1E2Sol,Theta1E3Sol,Theta0Sol,xivec,alpha,mu,sigma,inout,nu,tennumber,outof, Order):
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
            R[i,j]=-(((alpha**3)/4)*Integrate((mu**(-1))*(curl(Theta1j)*Conj(curl(Theta1i))),mesh, order=2*(Order+1))).real
            I[i,j]=((alpha**3)/4)*Integrate(inout*nu*sigma*((Theta1j+Theta0j+xij)*(Conj(Theta1i)+Theta0i+xii)),mesh, order=2*(Order+1)).real
    R+=np.transpose(R-np.diag(np.diag(R))).real
    I+=np.transpose(I-np.diag(np.diag(I))).real
    return R, I























