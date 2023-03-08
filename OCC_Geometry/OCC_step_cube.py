from netgen.occ import *

"""
James Elgy - 2022:
Cube example for Netgen OCC geometry loaded from existing step geometry.
"""

material_name = ['cube']
sigma = [1e6]
mur = [1]
alpha = 0.001

geo = OCCGeometry(r'Tutorials/Examples/Example_10mm_cube.step')
cube = geo.shape.Move((-geo.shape.center.x, -geo.shape.center.y, -geo.shape.center.z))

cube.bc('default')
cube.mat(material_name[0])
cube.maxh = 1

box = Box(Pnt(-1000, -1000, -1000), Pnt(1000,1000,1000))
box.mat('air')
box.bc('outer')
box.maxh=1000

joined_object = Glue([box, cube])
nmesh = OCCGeometry(joined_object).GenerateMesh(meshsize.coarse)
nmesh.Save(r'VolFiles/OCC_step_cube.vol')