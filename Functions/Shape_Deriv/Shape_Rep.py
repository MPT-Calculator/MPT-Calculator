import numpy as np
from ngsolve import *
import scipy.sparse as sp
import gc
import os
from ..Saving.DictionaryList import *



def Shape_Rep(fes,fesmech,u1Truncated,u2Truncated,u3Truncated,inout,mu_inv,alpha,Sols,TensorArray_obj,TensorArray_ms,\
            bilinear_bonus_int_order,ordershape, Theta0Sol,xivec,sigma,Omega,Tolerance,Maxsteps,Solver,N0_obj,N0_ms):

    """
    Paul Ledger - 2026.
    Function to compute the shape derivative of J .

    1) Sets up linear form for dJ[v] for a mechanical test function v

    Args:
        fes: H(curl) finite element space for theta1, theta0
        fesmech: (H^1)^3 finite element space for v
        u1Truncated (_type_): Ndof x M complex left singular matrix for e_1. If ReducedSolve is False, then replace U with sparse identity of size Ndof.
        u2Truncated (_type_): Ndof x M complex left singular mactrix for e_2. If ReducedSolve is False, then replace U with sparse identity of size Ndof.
        u3Truncated (_type_): Ndof x M complex left singular matrix for e_3. If ReducedSolve is False, then replace U with sparse identity of size Ndof.
        inout (fem.CoefficientFunction): material coefficient function. 1 inside objects, 0 outside
        mu_inv (comp.GridFunction): grid function for 1/mu_r. Note that for material discontinuities aligning with vertices no interpolation is done.
        alpha (float): object size scaling
        Sols (np.ndarray): Ndof x nfreqs x 3 vector of solution coefficients.
        TensorArray_obj: TensorArray of current object
        TensorArray_ms: TensorArray of measured (target) object
        bilinear_bonus_int_order (int): Integration order for the bilinear forms
        ordershape: order of the mechancial representation
        Theta0Sol (np.ndarray): ndof x 3 array of theta0 solutions.
        xivec (list): 3x3 list of direction vectors
        sigma: coefficent function for sigma
        Omega: array of output POD frequencies

    Returns:

    """

    # Hmeasure
    Hmeasure=np.array((1,1,1))
    # Hexcite
    Hexcite=np.array((1,1,1))


    gfi=GridFunction(fes)
    gf0i=GridFunction(fes)
    gfj=GridFunction(fes)
    gf0j=GridFunction(fes)

    # Test and trial functions
    umech = fesmech.TrialFunction()
    vmech = fesmech.TestFunction()

    dim1,dim2=np.shape(u1Truncated)
    U=np.zeros((3,dim1,dim2),dtype=complex)
    U[0,:,:]=u1Truncated[:,:]
    U[1,:,:]=u2Truncated[:,:]
    U[2,:,:]=u3Truncated[:,:]

    dJ=LinearForm(fesmech)

    # We want to loop over frequencies
    print(len(Omega),Sols.shape[1])

    Mu0=4.*np.pi*1e-7
    for k in range(Sols.shape[1]):

        nu=Omega[k]*Mu0*alpha**2 # without sigma

        # Set up MPTs of object and measurements
        MPTobj=np.zeros((3,3),dtype=complex)
        MPTms=np.zeros((3,3),dtype=complex)

        for i in range(3):
            for j in range(3):
                MPTobj[i,j]=TensorArray_obj[k,j+i*3]-N0_obj[i,j]
                MPTms[i,j]=TensorArray_ms[k,j+i*3]-N0_ms[i,j]

        # Compute the term sum_i, sum_j Hmeasure(i)@(MPT-MPTms)@Hexcite(j)
        term=0.
        term+=Hmeasure@((MPTobj[:,:]-MPTms[:,:])@Hexcite)

        # For every frequency we want to construct the grid functions

        for i in range(3):
            gi = np.squeeze(Sols[:,k, i])

            # This is the grid function representing theta_i^(1) computed from POD at frequency omega(k)
            gfi.vec.FV().NumPy()[:] = U[i,:, :]@gi
            gf0i.vec.FV().NumPy()[:]=Theta0Sol[:, i]

            for j in range(3):
                gj = np.squeeze(Sols[:,k, j])
                # This is the grid function representing theta_j^(1) computed from POD at frequency omega(k)
                gfj.vec.FV().NumPy()[:] = U[j,:, :]@gj
                gf0j.vec.FV().NumPy()[:]=Theta0Sol[:, j]

                # add contribution to shape derivative term (for the real part)
                sizefactor=-alpha**3/4
                dJ+=SymbolicLFI(2*np.real(term*Hmeasure[i]*(sizefactor*inout * mu_inv *InnerProduct(curl(gfi),2*Sym(Grad(vmech))* curl(gfj)))*Hexcite[j]),bonus_intorder=bilinear_bonus_int_order+ordershape)
                dJ+=SymbolicLFI(-2*np.real(term*Hmeasure[i]*(sizefactor*inout * mu_inv *InnerProduct(curl(gfi), curl(gfj)))*div(vmech)*Hexcite[j]),bonus_intorder=bilinear_bonus_int_order+ordershape)

                dJ+=SymbolicLFI(2*np.real(term*Hmeasure[i]*(sizefactor*(1-inout) * InnerProduct(curl(gfi),2*Sym(Grad(vmech))* curl(gfj)))*Hexcite[j]),bonus_intorder=bilinear_bonus_int_order+ordershape)
                dJ+=SymbolicLFI(-2*np.real(term*Hmeasure[i]*(sizefactor*(1-inout) * InnerProduct(curl(gfi), curl(gfj)))*div(vmech)*Hexcite[j]),bonus_intorder=bilinear_bonus_int_order+ordershape)


                # add contribution to shape derivative (for the imaginary part)
                sizefactor=alpha**3/4
                # nb nu defined without sigma above
                dJ+=SymbolicLFI(-2*np.real(term*Hmeasure[i]*(sizefactor*inout * 1j*sigma*nu *InnerProduct(gfi+xivec[i]+gf0i,2*Sym(Grad(vmech))* (gfj+xivec[j]+gf0j)))*Hexcite[j]),bonus_intorder=bilinear_bonus_int_order+ordershape)
                dJ+=SymbolicLFI(2*np.real(term*Hmeasure[i]*(sizefactor*inout * 1j*sigma*nu *InnerProduct(gfi+xivec[i]+gf0i,(gfj+xivec[j]+gf0j)))*Hexcite[j]*div(vmech)),bonus_intorder=bilinear_bonus_int_order+ordershape)


    shape = GridFunction(fesmech)
    shape.Set((0, 0, 0), BND)

    # Assemble (elastic type) system to give a representation of
    a =BilinearForm(fesmech, symmetric=True, condense=True)
    a += SymbolicBFI(InnerProduct(Sym(Grad(umech)),Sym(Grad(vmech)))+InnerProduct(umech,vmech), bonus_intorder=bilinear_bonus_int_order)


    # Solve the linear problem
    Solver="bddc"
    if Solver == "bddc":
        c = Preconditioner(a, "bddc")  # Apply the bddc preconditioner
    with TaskManager():
        a.Assemble()
        dJ.Assemble()

        print("Assembled linar form for shape derivative")


        if Solver == "local":
            c = Preconditioner(a, "local")  # Apply the local preconditioner
        c.Update()

        # Solve the problem
        dJ.vec.data += a.harmonic_extension_trans * dJ.vec
        res = dJ.vec.CreateVector()
        res.data = dJ.vec - a.mat * shape.vec
        inverse = CGSolver(a.mat, c.mat, precision=Tolerance, maxsteps=Maxsteps, printrates=True)
        shape.vec.data += inverse * res
        shape.vec.data += a.harmonic_extension * shape.vec
        shape.vec.data += a.inner_solve * dJ.vec



    return shape

def vtk_shape_output(shape, dom_nrs_metal,mur,sig,alpha,Omega,numelements,Order,mesh,Object,sweepname):

    print(' creating vtk output', end='\r')
    Refine=False
    Sols = []
    Sols.append(dom_nrs_metal)
    Sols.append(shape)

    # Creating Save Name:
    strmur = DictionaryList(mur, False)
    strsig = DictionaryList(sig, True)
    #savename = "Results/" + Object[:-4] + f"/al_{alpha}_mu_{strmur}_sig_{strsig}" + "/om_" + FtoS(Omega) + f"_el_{numelements}_ord_{Order}/Data/"
    savename='Results/' + sweepname + '/Data/'

    if Refine == True:
        vtk = VTKOutput(ma=mesh, coefs=Sols,
                            names=["Object", "Shape"], filename=savename + Object[:-4], subdivision=3)
    else:
        vtk = VTKOutput(ma=mesh, coefs=Sols,
                            names=["Object", "Shaoe"], filename=savename + Object[:-4], subdivision=0)
    vtk.Do()

        # Compressing vtk output and sending to zip file:
    zipObj = ZipFile(savename + 'VTU.zip', 'w', ZIP_DEFLATED)
    zipObj.write(savename + Object[:-4] + '.vtu', os.path.basename(savename + Object[:-4] + '.vtu'))
    zipObj.close()
    os.remove(savename + Object[:-4] + '.vtu')
    print(' vtk output created')

    return
