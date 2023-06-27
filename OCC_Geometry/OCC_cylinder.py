from netgen.occ import *
from ngsolve import *

"""
James Elgy - 2022:
Cylinder example for Netgen OCC geometry mesh generation.
Object has additional prismatic layers.
"""

# Setting mur, sigma, and defining the top level object name:
material_name = ['cylinder']
mur = [20]
sigma = [1e6]
alpha = 0.001

# Setting Boundary layer Options:
max_target_frequency = 1e8
boundary_layer_material = material_name[0]
number_of_layers = 2

# setting radius and height
r = 0.5
h = 25

# Generating OCC primative cylinder centered at [0,0,0] with radius r, height h, and orientated along X:
cylinder = Cylinder(Pnt(0,0,0), Y, r=r, h=h)

# Generating surrounding non-conducting region as [-1000,1000]^3 box:
box = Box(Pnt(-1000, -1000, -1000), Pnt(1000,1000,1000))

# setting material and bc names:
# For compatability, we want the non-conducting region to have the 'outer' boundary condition and be labeled as 'air'
cylinder.mat(material_name[0])
cylinder.bc('default')
box.mat('air')
box.bc('outer')

# Setting maxh:
cylinder.maxh = 1
box.maxh = 1000

# Joining the two meshes:
# Glue joins two OCC objects together without interior elemements
joined_object = Glue([cylinder, box])

# Generating Mesh:
geo = OCCGeometry(joined_object)
nmesh = geo.GenerateMesh()
# Applying Boundary Layers:
mu0 = 4 * 3.14159 * 1e-7
tau = (2/(max_target_frequency * sigma[0] * mu0 * mur[0]))**0.5 / alpha
layer_thicknesses = [(2**n)*tau for n in range(number_of_layers)]

nmesh.BoundaryLayer(boundary=".*", thickness=layer_thicknesses, material=boundary_layer_material,
                           domains=boundary_layer_material, outside=False)

nmesh.Save(r'VolFiles/OCC_cylinder.vol')

