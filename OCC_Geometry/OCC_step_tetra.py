from netgen.occ import *

material_name = ['tetra']
sigma = [5.96 * 10**6]
mur = [1.5]

geo = OCCGeometry(r'StepFiles/tetra2.step')
tetra = geo.shape.Move((-geo.shape.center.x, -geo.shape.center.y, -geo.shape.center.z))

#cube = Box(Pnt(-5,-5,-5), Pnt(5,5,5))
tetra.bc('default')
tetra.mat(material_name[0])
tetra.maxh = 0.05

box = Box(Pnt(-1000, -1000, -1000), Pnt(1000,1000,1000))
box.mat('air')
box.bc('outer')
box.maxh=1000

joined_object = Glue([box, tetra])
nmesh = OCCGeometry(joined_object).GenerateMesh(meshsize.coarse)
nmesh.Save(r'VolFiles/OCC_step_tetra.vol')