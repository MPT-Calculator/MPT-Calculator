from netgen.occ import *
# from ngsolve import *
from netgen.meshing import BoundaryLayerParameters
import numpy as np

"""
Paul Ledger - 2025 
Created geometry by setting a list of profiles and using thru geometry
Added new boundary layer capability
"""



# Setting mur, sigma, alpha, and defining the top level object name:
material_name = ['mat1']
mur = [32]
sigma = [1e6]
alpha = 0.001

# Boundary Layer Settings: max frequency under consideration, the total number of prismatic layers and the material of each layer.
# Setting Boundary layer Options:
max_target_frequency = 1e8
boundary_layer_material = material_name[0]
number_of_layers = 2


basic = Box(Pnt(-1, -1, -1), Pnt(1,1,1))
rx=Sphere((1,0,0),1)
lx=Sphere((-1,0,0),1)
ry=Sphere((0,1,0),1)
ly=Sphere((0,-1,0),1)
rz=Sphere((0,0,-1),1)
lz=Sphere((0,0,1),1)

solid=basic+rx+lx+ry+ly+rz+lz

print(solid.mass)

# setting material and bc names:
# For compatability, we want the non-conducting region to have the 'outer' boundary condition and be labeled as 'air'
solid.bc('default')
solid.mat(material_name[0])
solid.maxh = 0.2

# Generating a large non-conducting region. For compatability with MPT-Calculator, we set the boundary condition to 'outer'
# and the material name to 'air'.
box = Box(Pnt(-1000, -1000, -1000), Pnt(1000,1000,1000))
box.mat('air')
box.bc('outer')
box.maxh=1000
box=box-solid

# Joining the two meshes:
# Glue joins two OCC objects together without interior elemements
joined_object = Glue([solid, box])

# Generating Mesh (updated to new call below):
#nmesh = OCCGeometry(joined_object).GenerateMesh()


# Creating Boundary Layer Structure:
mu0 = 4 * 3.14159 * 1e-7
tau = (2/(max_target_frequency * sigma[0] * mu0 * mur[0]))**0.5 / alpha
layer_thicknesses = [(2**n)*tau for n in range(number_of_layers)]

#nmesh.BoundaryLayer(boundary=".*", thickness=layer_thicknesses, material=boundary_layer_material,
#                           domains=boundary_layer_material, outside=False)

B = BoundaryLayerParameters(boundary=".*", thickness=layer_thicknesses, new_material=boundary_layer_material,
                           domain=boundary_layer_material, outside=False, disable_curving=False )
nmesh = OCCGeometry(joined_object).GenerateMesh(meshsize.coarse,boundary_layers=[B]) 

nmesh.Save(r'VolFiles/OCC_lego_smthbrick_prism_32.vol')
# print(nmesh.GetMaterial(2))
from ngsolve import *
mesh = Mesh(nmesh)
print(f'Materials = {mesh.GetMaterials()}')
