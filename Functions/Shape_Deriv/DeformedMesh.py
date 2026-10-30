from ngsolve import *
from netgen.meshing import MeshPoint, Pnt
import copy

def DeformedMesh(mesh, V, tau=1.0):
    """
    Create a new NGSolve mesh with coordinates

        x_new = x + tau*V(x)

    while retaining the topology and region information
    of the original mesh.
    """

    ngmesh = copy.deepcopy(mesh.ngmesh)

    # evaluate V before modifying the mesh
    newpoints = []

    for v in mesh.vertices:
        p = mesh[v].point
        d = V(mesh(*p))

        newpoints.append(
            tuple(p[i] + tau*d[i] for i in range(mesh.dim))
        )

    # update Netgen vertices
    for i, p in enumerate(newpoints):
        ngmesh[i+1] = MeshPoint(Pnt(*p))

    return Mesh(ngmesh)
