from netgen.occ import *
from ngsolve import *

"""
James Elgy - 2022:
torus example for Netgen OCC geometry mesh generation.
"""

# Setting mur, sigma, and defining the top level object name:
object_name = 'torus'
mur = 1
sigma = 1e6

# setting radius of tube and distance from origin to center of tube
r = 1
R = 2

# Generating OCC primative 2d circle centered at [0,0,0] with radius r:
circ = Face(Wire([Circle((R, 0, 0), Y, r)]))
tor = circ.Revolve(Axis((0, 0, 0), Z), 360)

# Generating surrounding non-conducting region as [-1000,1000]^3 box:
box = Box(Pnt(-1000, -1000, -1000), Pnt(1000, 1000, 1000))

# setting material and bc names:
# For compatability, we want the non-conducting region to have the 'outer' boundary condition and be labeled as 'air'
tor.mat(object_name)
tor.bc('default')
box.mat('air')
box.bc('outer')

# Setting maxh:
tor.maxh = 0.5
box.maxh = 1000

# Joining the two meshes:
# Glue joins two OCC objects together without interior elemements
joined_object = Glue([tor, box])

# Generating Mesh:
geo = OCCGeometry(joined_object)
nmesh = geo.GenerateMesh()
nmesh.Save(r'VolFiles/OCC_torus.vol')
