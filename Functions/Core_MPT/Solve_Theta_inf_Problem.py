# James Elgy - 04/05/2023

import numpy as np
from matplotlib import pyplot as plt
import multiprocessing
import tqdm
from ngsolve import *
from ..Core_MPT.Thetainf import *
from ..Core_MPT.imap_execution import *
from warnings import warn

def Solve_Theta_inf_Problem(Additional_Int_Order, CPUs, Maxsteps, Order, Solver, Tolerance, alpha, epsi, inout, mesh,
                          recoverymode, sweepname):
    """
    Paul Ledger - 2025
    Function to call and run the thetainf solver for MPT calculator. Note that this is intended as a general function,
    thus options such as recoverymode and sweepname may not be relevant in all cases and can be set to False.

    recoverymode now raises an error if Thetainf.npy is not found. This is to avoid mistakenly calculating Thetainf for
    an incorrect set of parameters and improve user safety.

    Parameters
    ----------
    Additional_Int_Order: int bonus integration order added to linear and bilinear forms.
    CPUs: number of cpus assigned to the problem. 1 runs through in sequential mode.
    Maxsteps: int max steps assigned to the CGSolver.
    Order: int order of basis functions assigned in fes.
    Solver: str for preconditioner name, e.g. 'bddc'
    Tolerance: float solver tolerance
    alpha: float object scaling alpha
    epsi: float numeric regularisation constant
    inout: CoefficientFunction 1 inside object 0 outside.
    mesh: NGsolve mesh for the object
    recoverymode: bool for if thetainf can be loaded from disk rather than recalculated. Used in POD modes.
    sweepname: str for the folder path used in recoverymode.

    Returns
    -------
    ThetainfSol,
    Thetainfi,
    Thetainfj,
    fes,
    ndof,
    evec
    """

    # Setup the finite element space
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]

    boundarylist=[]
    for boundaries in mesh.GetBoundaries():
        if boundaries not in boundarylist:
            boundarylist.append(boundaries)

    #Create string for applying Dirichlet to all boundaries
    Dirfullstring=""
    count=0
    for boundaries in boundarylist:
        if count==0:
            Dirfullstring=boundaries
        else:
            Dirfullstring+="|"+boundaries
        count+=1

    # exclude outer
    Dirnotouterstring=""
    count=0
    for boundaries in boundarylist:
        if boundaries != "outer":
            if count==0:
                Dirnotouterstring=boundaries
            else:
                Dirnotouterstring+="|"+boundaries
        count+=1

    # Note this fes is only defined in the free space region!
    #fes = HCurl(mesh, order=Order, dirichlet="outer", gradientdomains=dom_nrs_metal,definedon=mesh.Materials("air"))
    fes = HCurl(mesh, order=Order, dirichlet=Dirfullstring,  gradientdomains=dom_nrs_metal, definedon=mesh.Materials("air"))

    #fes = HCurl(mesh, order=Order, dirichlet="outer|default", gradientdomains=dom_nrs_metal,definedon=mesh.Materials("air"))
    # fes = HCurl(mesh, order=Order, dirichlet="outer", flags = { "nograds" : True })
    # Count the number of degrees of freedom
    ndof = fes.ndof
    # Define the vectors for the right hand side
    evec = [CoefficientFunction((1, 0, 0)), CoefficientFunction((0, 1, 0)), CoefficientFunction((0, 0, 1))]
    xivec = [CoefficientFunction((0, -z, y)), CoefficientFunction((z, 0, -x)), CoefficientFunction((-y, x, 0))]
    # Setup the grid functions and array which will be used to save
    Thetainfi = GridFunction(fes)
    Thetainfj = GridFunction(fes)
    ThetainfSol = np.zeros([ndof, 3])

    if CPUs > 1:
        if recoverymode is False:
            # Setup the inputs for the functions to run
            ThetainfCPUs = min(3, multiprocessing.cpu_count(), CPUs)
            Runlist = []
            for i in range(3):
                if Theta0CPUs < 3:
                    NewInput = (
                    fes, Order, alpha, inout, evec[i],xivec[i], Tolerance, Maxsteps, epsi, i + 1, Solver, Additional_Int_Order, mesh, Dirnotouterstring, 'Thetainf')
                else:
                    NewInput = (fes, Order, alpha, inout, evec[i],xivec[i], Tolerance, Maxsteps, epsi, "No Print", Solver,
                                Additional_Int_Order, mesh, Dirnotouterstring, 'Thetainf')
                Runlist.append(NewInput)
            # Run on the multiple cores
            with multiprocessing.get_context("spawn").Pool(ThetainfCPUs) as pool:
                Output = list(tqdm.tqdm(pool.map(imap_version, Runlist), total=len(Runlist), desc='Solving Thetainf'))

            print(' solved thetainf problems    ')

            # Unpack the outputs
            for i, Direction in enumerate(Output):
                ThetainfSol[:, i] = Direction
        else:
            ThetainfSol = np.load('Results/' + sweepname + '/Data/Thetainf.npy')

    else:
        if recoverymode is False:
            # Run in three directions and save in an array for later
            for i in tqdm.tqdm(range(3), desc='Solving Thetainf'):
                ThetainfSol[:, i] = Thetainf(fes, Order, alpha, inout, evec[i], xivec[i], Tolerance, Maxsteps, epsi, i + 1,
                                         Solver, Additional_Int_Order, mesh, Dirnotouterstring)
            print(' solved thetainf problems   ')
        else:
            ThetainfSol = np.load('Results/' + sweepname + '/Data/Thetainf.npy')


    return ThetainfSol, Thetainfi, Thetainfj, fes, ndof, evec


if __name__ == '__main__':
    pass
