from netgen.occ import *
from netgen.meshing import BoundaryLayerParameters

"""
Paul Ledger - 2025 
Added new boundary layer capability
"""

material_name = ['steel']
mur = [141.3135696662735]
sigma = [15000000.0]

# filename = r'/home/james/Documents/EInScan_3D_Scanner_Work/EinScan_Sept22/Key 4/Key_4_Low_Detail_simplified_remeshed.step'
filename = 'StepFiles/Key_4_Low_Detail_simplified_remeshed.step'
geo = OCCGeometry(filename)
geo.Heal()
geo = geo.shape.Move((-geo.shape.solids[0].center.x, -geo.shape.solids[0].center.y, -geo.shape.solids[0].center.z))

geo.bc('default')
geo.mat(material_name[0])
geo.maxh = 0.1

bounding_box = Box(Pnt(-1000, -1000, -1000), Pnt(1000, 1000, 1000))
bounding_box.mat('air')
bounding_box.bc('outer')
bounding_box.maxh=1000
bounding_box = bounding_box-geo

geo2 = OCCGeometry(Glue([geo, bounding_box]))
#nmesh = geo2.GenerateMesh(minh=5)
#nmesh.BoundaryLayer(boundary=".*", thickness=[5e-3], material=material_name[0],
#                           domains=material_name[0], outside=False)
B = BoundaryLayerParameters(boundary=".*", thickness=[5e-3], new_material=material_name[0],
                           domain=material_name[0], outside=False, disable_curving=False)

#B = BoundaryLayerParameters(boundary=".*", thickness=layer_thicknesses, new_material=boundary_layer_material,
#                           domain=boundary_layer_material, outside=False,  disable_curving=False)

#geo2.Heal() # Attempt to Heal geometry first
#nmesh = geo2.GenerateMesh(minh=5,boundary_layers=[B])
nmesh = geo2.GenerateMesh(minh=5)

nmesh.Save('VolFiles/OCC_test_key_4.vol')
