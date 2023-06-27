# James Elgy - 02/06/2023

import numpy as np
from matplotlib import pyplot as plt
from ngsolve import *
import netgen.meshing as ngmeshing
from ..MeshMaking.VolMatUpdater import *
import runpy

def check_mesh_volumes(mesh, inout, use_OCC, Object, integration_Order, curve):
    """
    James Elgy 2023:
    Adaption of CheckValid to compute mesh volume and in case of OCC geometry compare to exact known volumes. Note,
    that this does not scale by alpha

    Parameters
    ----------
    mesh - NGMesh object
    inout - CoefficientFunction - 1 inside, 0 outside,
    use_OCC - bool - if using OCC geometries
    Object - mesh file path.

    Returns
    -------

    """
    # integration_Order = max([3*(curve - 1), 2*(Order + 1)])

    # Grabbing materials and mesh:
    Materials, mur, sig, inorout, cond, ntags, tags = VolMatUpdater(Object[:-4] + '.vol', True)
    cond_coef = [cond[mat] for mat in mesh.GetMaterials() ]
    conductor = CoefficientFunction(cond_coef)



    mesh_volume = Integrate(inout, mesh, order=integration_Order)

    print("Predicted unit object volume is",mesh_volume)
    totalvolume=0.
    for n in range(ntags):
        # loop over the conductor elements
        print("considering conductor element",n,ntags,tags[n])
        volumepart = Integrate(myinout(conductor,n,ntags), mesh, order=integration_Order)
        if tags[n] != "air":
            totalvolume = totalvolume + volumepart
    print("Calculated conductor volume as sum",totalvolume)

    if use_OCC is True:
        out = runpy.run_path(f'OCC_Geometry/{Object[:-4]}.py')
        nmesh = out['nmesh']


# Helper function that returns 1 for if index=n and 0 otherwise
# for a mesh made of a general number of material tags
def myinout(index,n,ntags):
    """
    Helper function that returns 1 for if index=n and 0 otherwise for a mesh made of a general number of material tags
    """
    prod=1.
    den=1.
    for k in range(0,ntags+1):
        if k != n:
            prod = prod*(index-k)
            den = den*(n-k)
    return prod/den

