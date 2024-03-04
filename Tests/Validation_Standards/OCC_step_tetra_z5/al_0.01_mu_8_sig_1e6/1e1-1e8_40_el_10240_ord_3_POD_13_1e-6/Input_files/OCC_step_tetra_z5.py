from netgen.occ import *

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

joined_object = Glue([box, tetra])
nmesh = OCCGeometry(joined_object).GenerateMesh(meshsize.coarse)

delta = (2/(1e8*4*3.14159*1e-7*sigma[0]*mur[0]))**(0.5) / 0.001
nmesh.BoundaryLayer(boundary=".*", thickness=[delta, 2*delta], material=material_name[0],
                           domains=material_name[0], outside=False)


nmesh.Save(r'VolFiles/OCC_step_tetra_z5.vol')