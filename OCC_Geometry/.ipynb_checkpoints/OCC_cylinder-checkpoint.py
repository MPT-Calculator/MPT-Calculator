from netgen.occ import *
from ngsolve import *

"""
James Elgy - 2022:
Cylinder example for Netgen OCC geometry mesh generation.
"""



# Setting mur, sigma, and defining the top level object name:
object_name = 'cylinder'
mur = 1
sigma = 1e6

# setting radius and height
r = 1
h = 10

# Generating OCC primative cylinder centered at [0,0,0] with radius r, height h, and orientated along X:
cylinder = Cylinder(Pnt(0,0,0), X, r=r, h=h)

# Generating surrounding non-conducting region as [-1000,1000]^3 box:
box = Box(Pnt(-1000, -1000, -1000), Pnt(1000,1000,1000))

# setting material and bc names:
# For compatability, we want the non-conducting region to have the 'outer' boundary condition and be labeled as 'air'
cylinder.mat(object_name)
cylinder.bc('default')
box.mat('air')
box.bc('outer')

# Setting maxh:
cylinder.maxh = 0.5
box.maxh = 1000

# Joining the two meshes:
# Glue joins two OCC objects together without interior elemements
joined_object = Glue([cylinder, box])

# Generating Mesh:
geo = OCCGeometry(joined_object)
nmesh = geo.GenerateMesh()
nmesh.Save(r'VolFiles/OCC_cylinder.vol')

