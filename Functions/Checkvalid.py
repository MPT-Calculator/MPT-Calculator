#This file contains the function for checking the validity of the eddy current model
#Functions -PODP

#Importing
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import scipy.linalg as slin
import multiprocessing as multiprocessing
import netgen.meshing as ngmeshing
import sys
from ngsolve import *

sys.path.insert(0,"Functions")
sys.path.insert(0,"Settings")
from Settings import SolverParameters


def Checkvalid(Object,Order,alpha,inorout,mur,sig):
    Object = Object[:-4]+".vol"
    #Set order Ordercheck to be of low order to speed up computation.
    Ordercheck = 1 
    #Accuracy is increased by increaing noutput, but at greater cost
    noutput=20

    #Set up the Solver Parameters
    Solver,epsi,Maxsteps,Tolerance = SolverParameters()

    #Loading the object file
    ngmesh = ngmeshing.Mesh(dim=3)
    ngmesh.Load("VolFiles/"+Object)

    #Creating the mesh and defining the element types
    mesh = Mesh("VolFiles/"+Object)
    mesh.Curve(5)#This can be used to refine the mesh

    #Set materials
    mu_coef = [ mur[mat] for mat in mesh.GetMaterials() ]
    mu = CoefficientFunction(mu_coef)
    inout_coef = [inorout[mat] for mat in mesh.GetMaterials() ]
    inout = CoefficientFunction(inout_coef)
    sigma_coef = [sig[mat] for mat in mesh.GetMaterials() ]
    sigma = CoefficientFunction(sigma_coef)

    #Scalars
    Mu0 = 4*np.pi*10**(-7)

    #Setup the finite element space
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]

    femfull = H1(mesh, order=Ordercheck,dirichlet="default|outside")
    freedofs = femfull.FreeDofs()

    ndof=femfull.ndof
    Output = np.zeros([ndof,noutput],dtype=float)
    Averg = np.zeros([1,3],dtype=float)

    # we want to create a list of coordinates where we would like to apply BCs
    list=np.zeros([noutput,3],dtype=float)
    npp=0
    for el in mesh.Elements(BND):

        if el.mat == "default":
            Averg[0,:]=0
            #determine the average coordinate
            for v in el.vertices:
                Averg[0,:]+=mesh[v].point[:]
            Averg=Averg/3
            if npp < noutput:
                list[npp,:]=Averg[0,:]
                npp+=1

    print(" solving problems", end='\r')
    sval=(Integrate(inout, mesh))**(1/3)
    for i in range(noutput):
        sol=GridFunction(femfull)
        sol.Set(exp(-((x-list[i,0])**2 + (y-list[i,1])**2 + (z-list[i,2])**2)/sval**2),definedon=mesh.Boundaries("default"))

        u = femfull.TrialFunction()
        v = femfull.TestFunction()

    	# the bilinear-form
        a = BilinearForm(femfull, symmetric=True,  condense=True)
        a += 1/alpha**2*grad(u)*grad(v)*dx
        a += u*v*dx

		# the right hand side
        f = LinearForm(femfull)
        f += 0 * v * dx

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
        res.data = f.vec - a.mat * sol.vec
        inverse= CGSolver(a.mat, c.mat, precision=Tolerance, maxsteps=Maxsteps)
        sol.vec.data += inverse  * res
        sol.vec.data += a.inner_solve * f.vec
        sol.vec.data += a.harmonic_extension * sol.vec

        Output[:,i] = sol.vec.FV().NumPy()
    print(" problems solved        ")
    Mc = np.zeros([noutput,noutput],dtype=float)
    M0 = np.zeros([noutput,noutput],dtype=float)

    print(" computing matrices", end='\r')
    # create numpy arrays by passing solutions back to NG Solve
    Soli=GridFunction(femfull)
    Solj=GridFunction(femfull)
    
    for i in range(noutput):
        Soli.Set(exp(-((x-list[i,0])**2 + (y-list[i,1])**2 + (z-list[i,2])**2)/sval**2),definedon=mesh.Boundaries("default"))
        Soli.vec.FV().NumPy()[:]=Output[:,i]

        for j in range(i,noutput):
            Solj.Set(exp(-((x-list[j,0])**2 + (y-list[j,1])**2 + (z-list[j,2])**2)/sval**2),definedon=mesh.Boundaries("default"))
            Solj.vec.FV().NumPy()[:]=Output[:,j]
            
            Mc[i,j] = Integrate(inout * (InnerProduct(grad(Soli),grad(Solj))/alpha**2+ InnerProduct(Soli,Solj)),mesh)
            Mc[j,i] = Mc[i,j]
            M0[i,j] = Integrate((1-inout) * (InnerProduct(grad(Soli),grad(Solj))/alpha**2+ InnerProduct(Soli,Solj)),mesh)
            M0[j,i] = M0[i,j]
    print(" matrices computed       ")
    
	# solve the eigenvalue problem
    print(" solving eigenvalue problem", end='\r')
    out=slin.eig(Mc+M0,Mc,left=False, right=False)
    print(" eigenvalue problem solved    ")

	# compute contants
    etasq = np.max((out.real))
    C = 1 # It is not clear what this value is.
    C1 = C * ( 1 + np.sqrt(etasq) )**2
    C2 = 2 * etasq

    epsilon = 8.854*10**-12
    sigmamin = Integrate(inout * sigma, mesh)/Integrate(inout, mesh)
    mumax = Integrate(inout * mu * Mu0, mesh)/Integrate(inout, mesh)
    volume = Integrate(inout, mesh)
    D = (volume * alpha**3)**(1/3)
    cond1 = np.sqrt(1/epsilon/mumax/D**2/C1)
    cond2 = 1/epsilon*sigmamin/C2
    cond = min(cond1,cond2)

    print(" maximum recomeneded frequency is ",str(round(cond/100)))
    
    return cond/100
