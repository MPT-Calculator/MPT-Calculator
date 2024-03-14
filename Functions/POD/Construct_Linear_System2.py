import numpy as np
from ngsolve import *
import scipy.sparse as sp
import gc

def Construct_Linear_System(Additional_Int_Order, BigProblem, Mu0, Theta0Sol, alpha, epsi, fes, fes2, inout, mu_inv, sigma,
                  xivec, NumSolverThreads, drop_tol, u1Truncated, u2Truncated, u3Truncated, dom_nrs_metal, PODErrorBars):
    
    # print(help(a0.mat.COO))
    # a0 = BilinearForm(fes2, symmetric=True, bonus_intorder=Additional_Int_Order, delete_zero_elements =drop_tol,keep_internal=False, symmetric_storage=True)
    # a0 += SymbolicBFI((mu_inv) * InnerProduct(curl(u), curl(v)), bonus_intorder=Additional_Int_Order)
    # a0 += SymbolicBFI((1j) * (1 - inout) * epsi * InnerProduct(u, v), bonus_intorder=Additional_Int_Order)
    
    # use_Assembly = True
    
    # Preallocation
    print(' creating reduced order model', end='\r')
    if NumSolverThreads != 'default':
        SetNumThreads(NumSolverThreads)    
    # Mu0=4*np.pi*10**(-7)
    nu_no_omega = Mu0 * (alpha ** 2)
    Theta_0 = GridFunction(fes)
    u, v = fes2.TnT()
    ndof2 = fes2.ndof
    cutoff = u1Truncated.shape[1]

    if PODErrorBars is True:
        fes0 = HCurl(fes2.mesh, order=0, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
        ndof0 = fes0.ndof
        RerrorReduced1 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)
        RerrorReduced2 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)
        RerrorReduced3 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)
        ProH = GridFunction(fes2) # ProH and ProL are overwritten and reused for all 3 directions
        ProL = GridFunction(fes0)
        
    else:
        RerrorReduced1 = None
        RerrorReduced2 = None
        RerrorReduced3 = None
        fes0 = None
        ndof0 = None
        ProL = None
        

    # Working on linear forms R1, R2, and R3
    Theta_0 = GridFunction(fes2)

    Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, 0]
    r1 = LinearForm(fes2)
    r1 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0, v), bonus_intorder=Additional_Int_Order)
    r1 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[0], v), bonus_intorder=Additional_Int_Order)
    r1.Assemble()
    R1 = r1.vec.FV().NumPy()
    HR1 = (np.conjugate(np.transpose(u1Truncated)) @ np.transpose(R1))
    
    if PODErrorBars is True:
        ProH.vec.FV().NumPy()[:] = R1
        ProL.Set(ProH)
        RerrorReduced1[:, 0] = ProL.vec.FV().NumPy()[:]
        
    del R1, r1


    Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, 1]
    r2 = LinearForm(fes2)
    r2 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0, v), bonus_intorder=Additional_Int_Order)
    r2 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[1], v), bonus_intorder=Additional_Int_Order)
    r2.Assemble()
    R2 = r2.vec.FV().NumPy()
    HR2 = (np.conjugate(np.transpose(u2Truncated)) @ np.transpose(R2))
    
    if PODErrorBars is True:
        ProH.vec.FV().NumPy()[:] = R2
        ProL.Set(ProH)
        RerrorReduced2[:, 0] = ProL.vec.FV().NumPy()[:]
    
    del R2, r2
    
    
    Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, 2]
    r3 = LinearForm(fes2)
    r3 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0, v), bonus_intorder=Additional_Int_Order)
    r3 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[2], v), bonus_intorder=Additional_Int_Order)
    r3.Assemble()
    R3 = r3.vec.FV().NumPy()
    HR3 = (np.conjugate(np.transpose(u3Truncated)) @ np.transpose(R3))
    
    if PODErrorBars is True:
        ProH.vec.FV().NumPy()[:] = R3
        ProL.Set(ProH)
        RerrorReduced3[:, 0] = ProL.vec.FV().NumPy()[:]
        
    del R3, r3



    # Working on A0
    a0 = BilinearForm(fes2, symmetric=True, bonus_intorder=Additional_Int_Order, delete_zero_elements =drop_tol,keep_internal=False, symmetric_storage=True)
    a0 += SymbolicBFI((mu_inv) * InnerProduct(curl(u), curl(v)), bonus_intorder=Additional_Int_Order)
    a0 += SymbolicBFI((1j) * (1 - inout) * epsi * InnerProduct(u, v), bonus_intorder=Additional_Int_Order)
    
    if BigProblem is False:
        with TaskManager():
            a0.Assemble()
        rows, cols, vals = a0.mat.COO()
        A0sym = sp.csr_matrix((vals, (rows, cols)),shape=(ndof2,ndof2))
        del rows,cols,vals, a0
        gc.collect()
        A0 = A0sym + A0sym.T - sp.diags(A0sym.diagonal())
        del A0sym
    
    # Compute smaller matrices (U^m_i)^H A0 U^m_i. Note that a0 is not fully assembled.
    
    if BigProblem is True:
        #E1
        A0H = np.zeros([ndof2, cutoff], dtype=complex)
        read_vec = GridFunction(fes2).vec.CreateVector()
        write_vec = GridFunction(fes2).vec.CreateVector()
        
        for i in range(cutoff):
            read_vec.FV().NumPy()[:] = u1Truncated[:, i]
            with TaskManager():
                a0.Apply(read_vec, write_vec)
            A0H[:, i] = write_vec.FV().NumPy()
        HA0H1 = (np.conjugate(np.transpose(u1Truncated)) @ A0H)
        
        if PODErrorBars is True:
            for i in range(cutoff):
                ProH.vec.FV().NumPy()[:] = A0H[:, i]
                ProL.Set(ProH)
                RerrorReduced1[:, i + 1] = ProL.vec.FV().NumPy()[:]
        
        #E2
        for i in range(cutoff):
            read_vec.FV().NumPy()[:] = u2Truncated[:, i]
            with TaskManager():
                a0.Apply(read_vec, write_vec)
            A0H[:, i] = write_vec.FV().NumPy()
        HA0H2 = (np.conjugate(np.transpose(u2Truncated)) @ A0H)
        
        if PODErrorBars is True:
            for i in range(cutoff):
                ProH.vec.FV().NumPy()[:] = A0H[:, i]
                ProL.Set(ProH)
                RerrorReduced2[:, i + 1] = ProL.vec.FV().NumPy()[:]
        
        #E3
        for i in range(cutoff):
            read_vec.FV().NumPy()[:] = u3Truncated[:, i]
            with TaskManager():
                a0.Apply(read_vec, write_vec)
            A0H[:, i] = write_vec.FV().NumPy()
        HA0H3 = (np.conjugate(np.transpose(u3Truncated)) @ A0H)
        
        if PODErrorBars is True:
            for i in range(cutoff):
                ProH.vec.FV().NumPy()[:] = A0H[:, i]
                ProL.Set(ProH)
                RerrorReduced3[:, i + 1] = ProL.vec.FV().NumPy()[:]

        # ao and A0H are no longer needed.
        del a0, A0H
    else:
        
        if PODErrorBars is True:
            A0H = A0 @ u1Truncated
            for i in range(cutoff):
                ProH.vec.FV().NumPy()[:] = A0H[:, i]
                ProL.Set(ProH)
                RerrorReduced1[:, i + 1] = ProL.vec.FV().NumPy()[:]
                
            A0H = A0 @ u2Truncated
            for i in range(cutoff):
                ProH.vec.FV().NumPy()[:] = A0H[:, i]
                ProL.Set(ProH)
                RerrorReduced2[:, i + 1] = ProL.vec.FV().NumPy()[:]
                
            A0H = A0 @ u3Truncated
            for i in range(cutoff):
                ProH.vec.FV().NumPy()[:] = A0H[:, i]
                ProL.Set(ProH)
                RerrorReduced3[:, i + 1] = ProL.vec.FV().NumPy()[:]
            del A0H
        
        HA0H1 = np.conjugate(np.transpose(u1Truncated)) @ A0 @u1Truncated
        HA0H2 = np.conjugate(np.transpose(u2Truncated)) @ A0 @u2Truncated
        HA0H3 = np.conjugate(np.transpose(u3Truncated)) @ A0 @u3Truncated
        del A0

    
    
    # Working on A1
    a1 = BilinearForm(fes2, symmetric=True, delete_zero_elements =drop_tol,keep_internal=False, symmetric_storage=True)
    a1 += SymbolicBFI((1j) * inout * nu_no_omega * sigma * InnerProduct(u, v), bonus_intorder=Additional_Int_Order)
    
    if BigProblem is False:
        with TaskManager():
            a1.Assemble()
        rows, cols, vals = a1.mat.COO()
        A1sym = sp.csr_matrix((vals, (rows, cols)),shape=(ndof2,ndof2))
        del rows,cols,vals, a1
        gc.collect()
        A1 = A1sym + A1sym.T - sp.diags(A1sym.diagonal())
        del A1sym
    
    A1H = np.zeros([ndof2, cutoff], dtype=complex)

    if BigProblem is True:
        # Compute smaller matrices (U^m_i)^H A1 U^m_i.
        #E1
        for i in range(cutoff):
            read_vec.FV().NumPy()[:] = u1Truncated[:, i]
            with TaskManager():
                a1.Apply(read_vec, write_vec)
            A1H[:, i] = write_vec.FV().NumPy()
        HA1H1 = (np.conjugate(np.transpose(u1Truncated)) @ A1H)
        
        if PODErrorBars == True:
            for i in range(cutoff):
                ProH.vec.FV().NumPy()[:] = A1H[:, i]
                ProL.Set(ProH)
                RerrorReduced1[:, i + cutoff + 1] = ProL.vec.FV().NumPy()[:]
        
        #E2
        for i in range(cutoff):
            read_vec.FV().NumPy()[:] = u2Truncated[:, i]
            with TaskManager():
                a1.Apply(read_vec, write_vec)
            A1H[:, i] = write_vec.FV().NumPy()
        HA1H2 = (np.conjugate(np.transpose(u2Truncated)) @ A1H)

        if PODErrorBars == True:
            for i in range(cutoff):
                ProH.vec.FV().NumPy()[:] = A1H[:, i]
                ProL.Set(ProH)
                RerrorReduced2[:, i + cutoff + 1] = ProL.vec.FV().NumPy()[:]
        
        #E3
        for i in range(cutoff):
            read_vec.FV().NumPy()[:] = u3Truncated[:, i]
            with TaskManager():
                a1.Apply(read_vec, write_vec)
            A1H[:, i] = write_vec.FV().NumPy()
        HA1H3 = (np.conjugate(np.transpose(u3Truncated)) @ A1H)
        
        if PODErrorBars == True:
            for i in range(cutoff):
                ProH.vec.FV().NumPy()[:] = A1H[:, i]
                ProL.Set(ProH)
                RerrorReduced3[:, i + cutoff + 1] = ProL.vec.FV().NumPy()[:]
        
        
        # a1 and A1H are no longer needed.
        del a1, A1H, read_vec, write_vec
    else:
        
        if PODErrorBars is True:
            A1H = A1 @ u1Truncated
            for i in range(cutoff):
                ProH.vec.FV().NumPy()[:] = A1H[:, i]
                ProL.Set(ProH)
                RerrorReduced1[:, i + 1] = ProL.vec.FV().NumPy()[:]
                
            A1H = A1 @ u2Truncated
            for i in range(cutoff):
                ProH.vec.FV().NumPy()[:] = A1H[:, i]
                ProL.Set(ProH)
                RerrorReduced2[:, i + 1] = ProL.vec.FV().NumPy()[:]
                
            A1H = A1 @ u3Truncated
            for i in range(cutoff):
                ProH.vec.FV().NumPy()[:] = A1H[:, i]
                ProL.Set(ProH)
                RerrorReduced3[:, i + 1] = ProL.vec.FV().NumPy()[:]
            del A1H
            
        HA1H1 = np.conjugate(np.transpose(u1Truncated)) @ A1 @u1Truncated
        HA1H2 = np.conjugate(np.transpose(u2Truncated)) @ A1 @u2Truncated
        HA1H3 = np.conjugate(np.transpose(u3Truncated)) @ A1 @u3Truncated
        del A1

    

    return HA0H1, HA0H2, HA0H3, HA1H1, HA1H2, HA1H3, HR1, HR2, HR3, ProL, RerrorReduced1, RerrorReduced2, RerrorReduced3, fes0, ndof0
    
    # # Comparison
    # a0.Assemble()
    # rows, cols, vals = a0.mat.COO()
    # A0sym = sp.csr_matrix((vals, (rows, cols)),shape=(ndof2,ndof2))
    # del rows,cols,vals, a0
    # gc.collect()
    # A0 = A0sym + A0sym.T - sp.diags(A0sym.diagonal())
    # del A0sym

    # A0H_2 = np.zeros([ndof2, cutoff], dtype=complex)
    # A0H_2 = A0@u1Truncated