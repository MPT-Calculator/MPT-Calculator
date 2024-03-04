from netgen.occ import *

material_name = ['steel']
mur = [141.3135696662735]
sigma = [15000000.0]

# filename = r'/home/james/Documents/EInScan_3D_Scanner_Work/EinScan_Sept22/Key 4/Key_4_Low_Detail_simplified_remeshed.step'
filename = 'StepFiles/Key_4_Low_Detail_simplified_remeshed.step'
geo = OCCGeometry(filename)
geo = geo.shape.Move((-geo.shape.solids[0].center.x, -geo.shape.solids[0].center.y, -geo.shape.solids[0].center.z))

geo.bc('default')
geo.mat(material_name[0])
geo.maxh = 1000

bounding_box = Box(Pnt(-1000, -1000, -1000), Pnt(1000, 1000, 1000))
bounding_box.mat('air')
bounding_box.bc('outer')

geo2 = OCCGeometry(Glue([geo, bounding_box]))
nmesh = geo2.GenerateMesh(minh=5)
nmesh.BoundaryLayer(boundary=".*", thickness=[5e-3], material=material_name[0],
                           domains=material_name[0], outside=False)

nmesh.Save('VolFiles/OCC_test_key_4.vol')
