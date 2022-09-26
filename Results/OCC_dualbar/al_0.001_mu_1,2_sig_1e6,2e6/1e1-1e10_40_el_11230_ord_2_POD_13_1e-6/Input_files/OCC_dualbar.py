from netgen.occ import *
from ngsolve import *

"""
James Elgy - 2022:
Dual bar example for Netgen OCC geometry mesh generation.
This example illustated that lists can be used for multiple objects.
"""



# Setting mur, sigma, and defining the top level object name:
object_name = ['bar1', 'bar2']
mur = [1,2]
sigma = [1e6,2e6]


# Generating OCC primative boxes:
bar1 = Box(Pnt(-1,0,0), Pnt(0,1,1))
bar2 = Box(Pnt(0,0,0), Pnt(1,1,1))

# Generating surrounding non-conducting region as [-1000,1000]^3 box:
outer_box = Box(Pnt(-1000, -1000, -1000), Pnt(1000,1000,1000))

# setting material and bc names:
# For compatability, we want the non-conducting region to have the 'outer' boundary condition and be labeled as 'air'
bar1.mat(object_name[0])
bar2.mat(object_name[1])
bar1.bc('default')
bar2.bc('default')
outer_box.mat('air')
outer_box.bc('outer')

# Setting maxh:
bar1.maxh = 0.5
bar2.maxh = 0.5
outer_box.maxh = 1000

# Joining the two meshes:
# Glue joins two OCC objects together without interior elemements
joined_object = Glue([bar1, bar2, outer_box])

# Generating Mesh:
geo = OCCGeometry(joined_object)
nmesh = geo.GenerateMesh()
nmesh.Save(r'VolFiles/OCC_dualbar.vol')

