

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

sys.path.insert(0, "Functions")
from ..Core_MPT.Theta1 import *
from ..Core_MPT.Theta0 import *
from ..Core_MPT.MPTCalculator import *
from ..Core_MPT.imap_execution import *
from ..Saving.FtoS import *
from ..Saving.DictionaryList import *
from ..Core_MPT.MPT_Preallocation import *
from ..Core_MPT.Solve_Theta_0_Problem import *
from ..Core_MPT.Solve_Theta_inf_Problem import *
from ..Core_MPT.Calculate_N0 import *
from ..Core_MPT.Calculate_Minf import *
from ..Core_MPT.Theta0_Postprocessing import *
from ..Core_MPT.Mat_Method_Calc_Imag_Part import *
from ..Core_MPT.Mat_Method_Calc_Real_Part import *
from scipy import linalg
from scipy import sparse
from random import random

sys.path.insert(0, "Settings")
from Settings import SolverParameters, DefaultSettings
import gc

from Functions.Helper_Functions.count_prismatic_elements import count_prismatic_elements


def MPT_eigen(Object, Order, alpha, inorout, mur, sig, Omega, CPUs, VTK, Refine, Integration_Order, Additional_Int_Order, Order_L2, sweepname, drop_tol, fes, Theta0i, Theta_Return,
                    curve=5, theta_solutions_only=False, num_solver_threads='default'):

    _, Mu0, _, _, _, _,_, inout, mesh, mu_inv, numelements, sigma, bilinear_bonus_int_order = MPT_Preallocation([Omega], Object, [], curve, inorout,
                                                                                                                  mur, sig, Order, 0, sweepname,
                                                                                                                  num_solver_threads, drop_tol)
    # Set up the Solver Parameters
    Solver, epsi, Maxsteps, Tolerance, _, use_integral = SolverParameters()
    _,BigProblem,_,_,_, _, _, _tol = DefaultSettings()


    # define trial- and test-functions
    u = fes.TrialFunction()
    v = fes.TestFunction()

    # Weak form
    a = BilinearForm(fes,symmetric=True)#,condense=False)
    a += SymbolicBFI(mu_inv*curl(u)*curl(v),bonus_intorder=bilinear_bonus_int_order)


    # Compute sigma average
    sigma_avg=Integrate(inout*sigma,mesh)/Integrate(inout,mesh)

    # If sigma is homeogeneous then sigma/sigma_avg=1
    # if not this will weight contributions so that the projection still works for the eigen problem????

    # Weak form
    m = BilinearForm(fes,symmetric=True)#,condense=False)
    m += SymbolicBFI(inout*(sigma/sigma_avg)*u*v,bonus_intorder=bilinear_bonus_int_order) #bonus_intorder
    #m += SymbolicBFI(epsi*u*v*(1-inout),bonus_intorder=bilinear_bonus_int_order)
    #m += SymbolicBFI(0*u*v*(1-inout),bonus_intorder=bilinear_bonus_int_order)
    mext = BilinearForm(fes,symmetric=True)#,condense=False)
    mext += SymbolicBFI(u*v*(1-inout),bonus_intorder=bilinear_bonus_int_order)


    mreg = BilinearForm(fes,symmetric=True)#,condense=False)
    mreg += SymbolicBFI(inout*(sigma/sigma_avg)*u*v,bonus_intorder=bilinear_bonus_int_order) #bonus_intorder
    mreg += SymbolicBFI(epsi*u*v*(1-inout),bonus_intorder=bilinear_bonus_int_order)
    #m += SymbolicBFI(0*u*v*(1-inout),bonus_intorder=bilinear_bonus_int_order)

    apre = BilinearForm(fes,symmetric=True)#,condense=False)#BilinearForm(curl(u)*curl(v)*dx + 1*(1-inout)*u*v*dx+ reg*inout*u*v*dx)
    apre += SymbolicBFI(mu_inv*curl(u)*curl(v),bonus_intorder=bilinear_bonus_int_order)
    apre += SymbolicBFI(u*v*(sigma/sigma_avg)*inout,bonus_intorder=bilinear_bonus_int_order)
    apre += SymbolicBFI(epsi*u*v*(1-inout),bonus_intorder=bilinear_bonus_int_order)
    #apre += SymbolicBFI(0*u*v*(1-inout),bonus_intorder=bilinear_bonus_int_order)

    scale =2*np.pi*Integrate(1*inout,mesh)

    #pre = Preconditioner(apre, "direct", inverse="sparsecholesky")
    pre = Preconditioner(apre, "bddc")

    EigType="Dirichlet"# "Neumann"
    IterativeSolver="Iterative"# "Direct"


    with TaskManager():
        a.Assemble()
        mreg.Assemble()
        m.Assemble()
        apre.Assemble()
        pre.Update()
        mext.Assemble()
        pre.Update()

        # build gradient matrix as sparse matrix (and corresponding scalar FESpace)
        gradmat, fesh1 = fes.CreateGradient()


        gradmattrans = gradmat.CreateTranspose() # transpose sparse matrix
        math1 = gradmattrans @ mreg.mat @ gradmat   # multiply matrices

        # Try to build our own stiffness Matrix
        #u = fesh1.TrialFunction()
        #v = fesh1.TestFunction()
        #drop_tol=0#1e-3
        #k=BilinearForm(fesh1,symmetric=True,delete_zero_elements =drop_tol)

        #k+=SymbolicBFI(grad(u)*grad(v))
        #k.Assemble()
        #help(k.mat.Inverse(inverse="sparsecholesky"))
        #invh1=k.mat.Inverse(inverse="sparsecholesky")

        #help(math1)
        if EigType=="Neumann":
            math1[0,0] += 1 # Fix as discrete Laplacian is singular in this case.
            if IterativeSolver=="Direct":
                invh1 = math1.Inverse(inverse="sparsecholesky")
            else:

                math1smooth = math1.CreateSmoother()

                cgmath1 = CGSolver(
                    mat=math1,pre=math1smooth,
                    maxsteps=100,
                    precision=1e-8
                    )

                class H1Inverse(BaseMatrix):

                    def Mult(self, x, y):
                        y.data = cgmath1 * x#invh1s*x#cgH * x

                    def CreateColVector(self):
                        return math1.CreateColVector()

                    def CreateRowVector(self):
                        return math1.CreateRowVector()

                    def Height(self):
                        return math1.height

                    def Width(self):
                        return math1.width



                invh1 = H1Inverse()
        else:
            #math1[0,0] += 1     # fix the 1-dim kernel. not needed as discrete Laplacan is not singular.
            if IterativeSolver=="Direct":
                invh1 = math1.Inverse(inverse="sparsecholesky", freedofs=fesh1.FreeDofs()) # Note use of free DOFs only
            else:

                math1smooth = math1.CreateSmoother(freedofs=fesh1.FreeDofs())

                cgmath1 = CGSolver(
                    mat=math1,pre=math1smooth,
                    maxsteps=300,
                    precision=1e-8
                    )

                class H1Inverse(BaseMatrix):

                    def Mult(self, x, y):
                        y.data = cgmath1 * x#invh1s*x#cgH * x

                    def CreateColVector(self):
                        return math1.CreateColVector()

                    def CreateRowVector(self):
                        return math1.CreateRowVector()

                    def Height(self):
                        return math1.height

                    def Width(self):
                        return math1.width

                invh1 = H1Inverse()

#        help(math1.Inverse())
        #invh1 = math1.Inverse(inverse="bddc")
        # build the Poisson projector with operator Algebra:
        proj = IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ mreg.mat

        projpre = proj @ pre.mat

        #evals, evecs = solvers.PINVIT(a.mat, m.mat, pre=projpre, num=10, maxit=20) #150
        neval=1000#int(fes.ndof/40)#750#1000
        print("Looking for nevel modes=",neval)
        evals,evecs = myeigensolver(fes,a,m,mreg,mext,projpre,neval, mesh, Theta0i, Theta_Return,scale)

    return evals, evecs, sigma_avg

#def myeigensolver(fes,a,m,pre,num):
#    MaxIter=20
#    Tol=1e-3
#   Optimised version?
"""    u = GridFunction(fes, multidim=num)
    uvecs = MultiVector(u.vec, num)
    vecs = MultiVector(u.vec, 2*num)

    for v in vecs[0:num]:
        v.SetRandom()
    uvecs[:] = pre * vecs[0:num]
    lams = Vector(num * [1])
    ev=np.zeros(num)
    for i in range(MaxIter):
        vecs[0:num] = a.mat * uvecs - (m.mat * uvecs).Scale (lams)
        vecs[num:2*num] = pre * vecs[0:num]
        vecs[0:num] = uvecs

        vecs.Orthogonalize() # m.mat)

        asmall = InnerProduct (vecs, a.mat*vecs)
        msmall = InnerProduct (vecs, m.mat*vecs)
        evold=np.copy(ev[0:num])
        ev,evec = linalg.eigh(a=asmall, b=msmall)

        lams = Vector(ev[0:num])
        print (i, ":", [l for l in lams])

        uvecs[:] = vecs * Matrix(evec[:,0:num])
        if np.linalg.norm(ev[0:num]-evold)/np.linalg.norm(ev[0:num]) < Tol:
            break
    return lams,uvecs"""
"""r = u.vec.CreateVector()
    Av = u.vec.CreateVector()
    Mv = u.vec.CreateVector()

    vecs = []
    for i in range(2*num):
        vecs.append (u.vec.CreateVector())

    freedofs = fes.FreeDofs()
    for v in u.vecs:
        for i in range(len(u.vec)):
            v[i] = random() if freedofs[i] else 0


    lams = num * [1]

    asmall = Matrix(2*num, 2*num)
    msmall = Matrix(2*num, 2*num)

    for i in range(100):

        for j in range(num):
            vecs[j].data = u.vecs[j]
            r.data = a.mat * vecs[j] - lams[j] * m.mat * vecs[j]
            # r.data = 1/Norm(r) * r
            r *= 1/Norm(r)
            vecs[num+j].data = pre * r

        for j in range(2*num):
            Av.data = a.mat * vecs[j]
            Mv.data = m.mat * vecs[j]
            for k in range(2*num):
                asmall[j,k] = InnerProduct(Av, vecs[k])
                msmall[j,k] = InnerProduct(Mv, vecs[k])

        ev,evec = linalg.eigh(a=asmall, b=msmall)
        lams[0:num] = ev[0:num]
        print (i, ":", [lam for lam in lams])

        for j in range(num):
            r[:] = 0.0
            for k in range(2*num):
                r.data += float(evec[k,j]) * vecs[k]
            u.vecs[j].data = r"""




"""preconditioned inverse iteration"""
def myeigensolver(fes,a,m,mreg,mext, pre,num, mesh, Theta0i, Theta_Return,scale ):
    MaxIter=20
    Tol=5e-3
    printrates=True
    import scipy.linalg
    ndof=fes.ndof
    mata=a.mat
    matm=m.mat
    matmreg=mreg.mat
    matmext=mext.mat

    r = mata.CreateRowVector()

    uvecs = MultiVector(r, num)
    vecs = MultiVector(r, 2*num)
    # hv = MultiVector(r, 2*num)

    for v in vecs[0:num]:
        v.SetRandom()
    uvecs[:] = pre * vecs[0:num]
    lams = Vector(num * [1])#Vector(num * [1])
    ev=np.zeros(num)
    xivec = [CoefficientFunction((0, -z, y)), CoefficientFunction((z, 0, -x)), CoefficientFunction((-y, x, 0))]
    gfu = GridFunction(fes)
    xivecgfu = GridFunction(fes)
    Xivec_Return=np.zeros((fes.ndof,3))
    for i in range(3):
        xivecgfu.Set(xivec[i])
        Xivec_Return[:,i] = xivecgfu.vec.FV().NumPy()[:]
    print("Completed setting up xivecgfu")

    Filter=False
    Project=False
    for i in range(MaxIter):
        vecs[0:num] = mata * uvecs - (matm * uvecs).Scale (lams)
        vecs[num:2*num] = pre * vecs[0:num]
        vecs[0:num] = uvecs
        #vecs.Orthogonalize(matm)

        #if Project==True:
        #    vecs = projecttheta0(vecs, xivec, Theta0i, Theta_Return, gfu, mesh,  Xivec_Return, xivecgfu, matm, num,r  )

        #vecs[0:num] = uvecs

        vecs.Orthogonalize(matm)

        # hv[:] = mata * vecs
        # asmall = InnerProduct (vecs, hv)
        # hv[:] = matm * vecs
        # msmall = InnerProduct (vecs, hv)
        asmall = InnerProduct (vecs, mata * vecs)
        msmall = InnerProduct (vecs, matm * vecs)
        lamsold=list(lams)[0:num]
        #ev,evec = scipy.linalg.eigh(a=asmall, b=msmall)
        #print(msmall)
        ev,evec = scipy.linalg.eigh(a=asmall, b=msmall)

        if Filter==True and np.mod(i,3)==0: # Only apply filter every 4 iterations
            N=len(ev)
            lams = Vector(ev[0:N])
            uvecs[:] = vecs * Matrix(evec[:,0:N])
            lamsfilter, uvecsfilter, nkeep, filterarray = filtereigs(list(lams), uvecs, xivec, Theta0i, Theta_Return, gfu, mesh, num, r, pre, Xivec_Return, xivecgfu, matm, matmext )

            for n in range(len(lamsold)):
                if filterarray[n] != -1:
                    lamsold[n]=lamsold[filterarray[n]]

            num=nkeep
            uvecs = MultiVector(r, num)
            vecs = MultiVector(r, 2*num)
            lams=Vector(lamsfilter[0:num])
            uvecs[:]=uvecsfilter[0:num]
        else:
            lams = Vector(ev[0:num])
            if printrates:
                print (i, ":", list(lams))

            uvecs[:] = vecs * Matrix(evec[:,0:num])
            if Project==True:
                uvecs,lams,count = projecttheta0(uvecs, xivec, Theta0i, Theta_Return, gfu, mesh,  Xivec_Return, xivecgfu, matm, num,r,pre,lams,scale  )

            nkeep=num

        print(list(lams)[0:nkeep])
        lamslist=list(lams)[0:nkeep]
        # check convergence of first 40 modes
        if np.linalg.norm(np.array(lamslist)[0:40]-np.array(lamsold[0:40]))/np.linalg.norm(np.array(lamsold[0:40]))<Tol:
            lams=Vector(list(lams)[0:nkeep])
            break
        else:
            print("Res",np.linalg.norm(np.array(lamslist)[0:40]-np.array(lamsold[0:40]))/np.linalg.norm(np.array(lamsold[0:40])))
    print (i, ":", list(lams))
    if Filter==True:
        lams=Vector(list(lams)[0:nkeep])

    if Project==True:
        # remove last few (reset modes)
        for n in range(num-count-1, num):
            lams[n]=0
            uvecs[n]=np.zeros(ndof)

    uvecs.Orthogonalize(matm)
    return lams, uvecs

def filtereigs(lams, uvecs, xivec, Theta0i, Theta_Return, gfu, mesh, num, r, pre, Xivec_Return, xivecgfu, matm, matmext   ):

    N=len(lams)
    lamskeep=np.zeros(num)
    uvecskeep=MultiVector(r, num)
    nkeep=0
    TOL=1e-5
    filterarray=-1*np.ones(num, dtype=int)
    for n in range(num):

        gfu.vec.data = uvecs[n]

        flag=0
        for i in range(3):
            Theta0i.vec.FV().NumPy()[:] = Theta_Return[:, i]
            xivecgfu.vec.FV().NumPy()[:] = Xivec_Return[:, i]
            #I1= Integrate(gfu*(Theta0i+xivec[i]),mesh,definedon=mesh.Materials("object"))
            I = InnerProduct(gfu.vec,matm*(Theta0i.vec +xivecgfu.vec))
            S = InnerProduct(gfu.vec,matm*(gfu.vec))
            #Iext = InnerProduct(gfu.vec,matmext*(Theta0i.vec+xivecgfu.vec))# +xivecgfu.vec))
            #print(lams[n],i,I,Iext)
            #J= Integrate(gfu*(Theta0i+xivec[i]),mesh,definedon=mesh.Materials("air"))
            if np.abs(I/S) > TOL:
            #if np.abs(I/(S*lams[n])) > TOL:
                flag=1
            #print(n,lams[n],flag,I,J)
        if flag==1:
            # keep this mode
            lamskeep[nkeep]=lams[n]
            uvecskeep[nkeep]=uvecs[n]
            filterarray[nkeep]=n
            nkeep+=1

            if nkeep >= num:
                break# break loop if we have already found the number of eigen modes


    print("Out of ",num,"modes kept",nkeep)
    if nkeep <  num:
        print("Too few modes kept enlarge space")
        # add some extra random modes for the next iteration
        for n in range(nkeep,num):
            lamskeep[n]=num
        for u in uvecskeep[nkeep:num]:
            u.SetRandom()
        uvecs[0:nkeep] = uvecskeep[0:nkeep]
        uvecs[nkeep:num] = pre * uvecskeep[nkeep:num]
    else:
        uvecs[:]=uvecskeep[:]

    return lamskeep,uvecs, nkeep, filterarray


def projecttheta0(vecs, xivec, Theta0i, Theta_Return, gfu, mesh,  Xivec_Return, xivecgfu, matm, num, r,pre,lams,scale  ):

    TOL=5e-3
    ndof, dum = np.shape(Theta_Return)
    check=np.zeros((ndof))
    count=0
    for n in range(0,num):#(0,2*num):#(num,2*num):

        gfu.vec.data = vecs[n]
        #r.vec.data = gfu.vec.data
        flag=1
        for i in range(3):
            Theta0i.vec.FV().NumPy()[:] = Theta_Return[:, i]
            xivecgfu.vec.FV().NumPy()[:] = Xivec_Return[:, i]
            #I1= Integrate(gfu*(Theta0i+xivec[i]),mesh,definedon=mesh.Materials("object"))
            I = InnerProduct(gfu.vec,matm*(Theta0i.vec +xivecgfu.vec))
            #D = InnerProduct((Theta0i.vec +xivecgfu.vec),matm*(Theta0i.vec +xivecgfu.vec))
            #D=1
            #D=0.01**3*2*np.pi*1**3
            #I = InnerProduct(Theta0i.vec +xivecgfu.vec, matm*gfu.vec)
            D = InnerProduct(gfu.vec,matm*gfu.vec)

            #S = InnerProduct(gfu.vec,matm*(gfu.vec))
            #Iext = InnerProduct(gfu.vec,matmext*(Theta0i.vec+xivecgfu.vec))# +xivecgfu.vec))
            #print(lams[n],i,I,Iext)
            #J= Integrate(gfu*(Theta0i+xivec[i]),mesh,definedon=mesh.Materials("air"))

            # Project
            #check[:]= (I/D)*(Theta0i.vec.FV().NumPy()[:]+xivecgfu.vec.FV().NumPy()[:])
            # Check if < u,v>/<v,v> *v has norm greater than tolerance - if so keep.
            # otherwise flag for removal
            #print(n,np.abs(I/(D*scale)),D)
            if np.abs(I/D/scale)> TOL:
                flag =0

            #gfu.vec.data=  (I/D)*gfu.vec.data
            #gfu.vec.data=  (I/D)*(Theta0i.vec +xivecgfu.vec)
            #print(n-num,np.abs(I/D))
            #if np.abs(I/D) > TOL:
            #    flag=1
        if flag==1:
        #    # Throw away and generate a new one.
            print(n,np.abs(I/D/scale),D)
            vecs[n].SetRandom()
            vecs[n] = pre*vecs[n]
            lams[n]=100*lams[n]
            count+=1
    lamsarray=np.array(list(lams))
    order=np.argsort(lamsarray)
    lamsarray=lamsarray[order]
    lams=Vector(list(lamsarray))
    vecsnew = MultiVector(r, num)
    for n in range(num):
        vecsnew[n]=vecs[order[n]]

    return vecsnew,lams,count
