from netgen.occ import *

object_name = 'sphere'
sigma = 1e6
mur = 80

sphere = Sphere(Pnt(0,0,0), r=1)

pos_sphere = sphere - Box(Pnt(0,100,100), Pnt(-100,-100,-100))
neg_sphere = sphere - Box(Pnt(0,100,100), Pnt(100,-100,-100))
sphere = pos_sphere + neg_sphere

box = Box(Pnt(-1000,-1000,-1000), Pnt(1000,1000,1000))
box.bc('outer')
box.mat('air')

sphere.mat(object_name)
sphere.bc('default')
sphere.name = object_name
sphere.faces.name = object_name
sphere.maxh = 0.2


joined_object = Glue([box, sphere])
nmesh = OCCGeometry(joined_object).GenerateMesh(meshsize.coarse)
nmesh.BoundaryLayer(boundary=".*", thickness=[1e-3, 5e-3, 5e-2], material=object_name,
                    domains=object_name, outside=False)
nmesh.Save(r'VolFiles/OCC_sphere_prism.vol')