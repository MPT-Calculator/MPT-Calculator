from netgen.occ import *

material_name = ['sphere']
sigma = [1e6]
mur = [80]

sphere = Sphere(Pnt(0,0,0), r=1)

pos_sphere = sphere - Box(Pnt(0,100,100), Pnt(-100,-100,-100))
neg_sphere = sphere - Box(Pnt(0,100,100), Pnt(100,-100,-100))
sphere = pos_sphere + neg_sphere

box = Box(Pnt(-1000,-1000,-1000), Pnt(1000,1000,1000))
box.bc('outer')
box.mat('air')

sphere.mat(material_name[0])
sphere.bc('default')

sphere.maxh = 0.2


joined_object = Glue([box, sphere])
nmesh = OCCGeometry(joined_object).GenerateMesh(meshsize.coarse)
nmesh.BoundaryLayer(boundary=".*", thickness=[1e-3, 5e-3, 5e-2], material=material_name[0],
                    domains=material_name[0], outside=False)
nmesh.Save(r'VolFiles/OCC_sphere_prism.vol')