# James Elgy - 04/05/2023

import numpy as np
from matplotlib import pyplot as plt
import multiprocessing
import tqdm
from ngsolve import *
from ..Core_MPT.Theta0 import *
from ..Core_MPT.imap_execution import *
from warnings import warn

def Solve_Theta_0_Problem(Additional_Int_Order, CPUs, Maxsteps, Order, Solver, Tolerance, alpha, epsi, inout, mesh, mu_inv,
                          recoverymode, sweepname):
    """
    James Elgy - 2023
    Function to call and run the theta0 solver for MPT calculator. Note that this is intended as a general function,
    thus options such as recoverymode and sweepname may not be relevant in all cases and can be set to False.

    recoverymode now raises an error if Theta0.npy is not found. This is to avoid mistakenly calculating Theta0 for
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
    mu: CoefficientFunction with relative permeabilty assigned to each region
    recoverymode: bool for if theta0 can be loaded from disk rather than recalculated. Used in POD modes.
    sweepname: str for the folder path used in recoverymode.

    Returns
    -------
    Theta0Sol,
    Theta0i,
    Theta0j,
    fes,
    ndof,
    evec
    """

    # Setup the finite element space
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes = HCurl(mesh, order=Order, dirichlet="outer", gradientdomains=dom_nrs_metal)
    # fes = HCurl(mesh, order=Order, dirichlet="outer", flags = { "nograds" : True })
    # Count the number of degrees of freedom
    ndof = fes.ndof
    # Define the vectors for the right hand side
    evec = [CoefficientFunction((1, 0, 0)), CoefficientFunction((0, 1, 0)), CoefficientFunction((0, 0, 1))]
    # Setup the grid functions and array which will be used to save
    Theta0i = GridFunction(fes)
    Theta0j = GridFunction(fes)
    Theta0Sol = np.zeros([ndof, 3])

    if CPUs > 1:
        if recoverymode is False:
            # Setup the inputs for the functions to run
            Theta0CPUs = min(3, multiprocessing.cpu_count(), CPUs)
            Runlist = []
            for i in range(3):
                if Theta0CPUs < 3:
                    NewInput = (
                    fes, Order, alpha, mu_inv, inout, evec[i], Tolerance, Maxsteps, epsi, i + 1, Solver, Additional_Int_Order, 'Theta0')
                else:
                    NewInput = (fes, Order, alpha, mu_inv, inout, evec[i], Tolerance, Maxsteps, epsi, "No Print", Solver,
                                Additional_Int_Order, 'Theta0')
                Runlist.append(NewInput)
            # Run on the multiple cores
            with multiprocessing.get_context("spawn").Pool(Theta0CPUs) as pool:
                Output = list(tqdm.tqdm(pool.map(imap_version, Runlist), total=len(Runlist), desc='Solving Theta0'))

            print(' solved theta0 problems    ')

            # Unpack the outputs
            for i, Direction in enumerate(Output):
                Theta0Sol[:, i] = Direction
        else:
            Theta0Sol = np.load('Results/' + sweepname + '/Data/Theta0.npy')

    else:
        if recoverymode is False:
            # Run in three directions and save in an array for later
            for i in tqdm.tqdm(range(3), desc='Solving Theta0'):
                Theta0Sol[:, i] = Theta0(fes, Order, alpha, mu_inv, inout, evec[i], Tolerance, Maxsteps, epsi, i + 1,
                                         Solver, Additional_Int_Order)
            print(' solved theta0 problems   ')
        else:
            Theta0Sol = np.load('Results/' + sweepname + '/Data/Theta0.npy')


    return Theta0Sol, Theta0i, Theta0j, fes, ndof, evec


if __name__ == '__main__':
    pass
