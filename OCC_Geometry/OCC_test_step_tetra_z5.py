from netgen.occ import *
from netgen.meshing import BoundaryLayerParameters

"""
Paul Ledger - 2025 
Added new boundary layer capability
"""

material_name = ['tetra']
sigma = [1 * 10**6]
mur = [8]

geo = OCCGeometry(r'StepFiles/irregular_tetra_z=5.step')
tetra = geo.shape.Move((-geo.shape.center.x, -geo.shape.center.y, -geo.shape.center.z))

#cube = Box(Pnt(-5,-5,-5), Pnt(5,5,5))
tetra.bc('default')
tetra.mat(material_name[0])
tetra.maxh = 0.5

box = Box(Pnt(-1000, -1000, -1000), Pnt(1000,1000,1000))
box.mat('air')
box.bc('outer')
box.maxh=1000
box=box-tetra

joined_object = Glue([box, tetra])

delta = (2/(1e8*4*3.14159*1e-7*sigma[0]*mur[0]))**(0.5) / 0.001

B = BoundaryLayerParameters(boundary=".*", thickness=[delta, 2*delta], new_material=material_name[0],
                           domain=material_name[0], outside=False)

nmesh = OCCGeometry(joined_object).GenerateMesh(meshsize.coarse,boundary_layers=[B])

nmesh.Save(r'VolFiles/OCC_test_step_tetra_z5.vol')
