from netgen.occ import *
from ngsolve import *

"""
James Elgy - 2022:
sphere example for Netgen OCC geometry mesh generation.
"""



# Setting mur, sigma, and defining the top level object name:
material_name = ['steel']
mur = [1]
sigma = [1e6]
alpha = 1e-2

# setting radius
r = 1

# Generating OCC primative sphere centered at [0,0,0] with radius r:
sphere = Sphere(Pnt(0,0,0), r=r)

# Generating surrounding non-conducting region as [-1000,1000]^3 box:
box = Box(Pnt(-1000, -1000, -1000), Pnt(1000,1000,1000))

# setting material and bc names:
# For compatability, we want the non-conducting region to have the 'outer' boundary condition and be labeled as 'air'
sphere.mat(material_name[0])
sphere.bc('default')
box.mat('air')
box.bc('outer')

# Setting maxh:
sphere.maxh = 0.2
box.maxh = 1000

# Joining the two meshes:
# Glue joins two OCC objects together without interior elemements
joined_object = Glue([sphere, box])

# Generating Mesh:
geo = OCCGeometry(joined_object)
nmesh = geo.GenerateMesh()
nmesh.Save(r'VolFiles/OCC_sphere.vol')



