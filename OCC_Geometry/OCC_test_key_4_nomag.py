from netgen.occ import *
from netgen.meshing import BoundaryLayerParameters

"""
Paul Ledger - 2025 
Added new boundary layer capability (disabled due to meshing issues)
"""

material_name = ['steel']
mur = [4]#[141.3135696662735] # set to lowere mu_r as boundary layers are disabled
sigma = [15000000.0]

# filename = r'/home/james/Documents/EInScan_3D_Scanner_Work/EinScan_Sept22/Key 4/Key_4_Low_Detail_simplified_remeshed.step'
filename = 'StepFiles/Key_4_Low_Detail_simplified_remeshed.step'
geo = OCCGeometry(filename)
#geo.Heal() # did not improve situation
geo = geo.shape.Move((-geo.shape.solids[0].center.x, -geo.shape.solids[0].center.y, -geo.shape.solids[0].center.z))

geo.bc('default')
geo.mat(material_name[0])

bounding_box = Box(Pnt(-1000, -1000, -1000), Pnt(1000, 1000, 1000))
bounding_box.mat('air')
bounding_box.bc('outer')
bounding_box = bounding_box-geo

geo2 = OCCGeometry(Glue([geo, bounding_box]))

# Boundary layers diaabled for this test object
#B = BoundaryLayerParameters(boundary=".*", thickness=[5e-3], new_material=material_name[0],
#                           domain=material_name[0], outside=False, disable_curving=False)

#nmesh = geo2.GenerateMesh(minh=5,boundary_layers=[B])
nmesh = geo2.GenerateMesh()

nmesh.Save('VolFiles/OCC_test_key_4_nomag.vol')
