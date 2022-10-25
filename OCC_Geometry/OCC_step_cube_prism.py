from netgen.occ import *

material_name = ['cube']
sigma = 1e6
mur = 100

geo = OCCGeometry(r'tutorials/Examples/Example_10mm_cube.step')
cube = geo.shape.Move((-geo.shape.center.x, -geo.shape.center.y, -geo.shape.center.z))

#cube = Box(Pnt(-5,-5,-5), Pnt(5,5,5))
cube.bc('default')
cube.mat(material_name[0])
cube.solids.name = material_name[0]
cube.faces.name = 'default'
cube.maxh = 1

box = Box(Pnt(-1000, -1000, -1000), Pnt(1000,1000,1000))
box.mat('air')
box.bc('outer')
box.maxh=1000

joined_object = Glue([box, cube])
nmesh = OCCGeometry(joined_object).GenerateMesh(meshsize.coarse)
nmesh.BoundaryLayer(boundary=".*", thickness=[0.5], material=material_name[0], domains=material_name[0], outside=False)

nmesh.Save(r'VolFiles/OCC_step_cube_prism.vol')