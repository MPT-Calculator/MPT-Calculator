from netgen.occ import *
# from ngsolve import *
from netgen.meshing import BoundaryLayerParameters

"""
James Elgy - 2022:
sphere example for Netgen OCC geometry mesh generation.
Object has prismatic boundary layer elements added.

EDIT 2023:
Netgen-Mesher version 6.2.2301 gives a different result for the assigned materials when compared to version 6.2.2204.
The material assinged to 'box' should be 'air', and indeed this is what is reported when using the older version of
netgen. When using the new version, it reports the material as 'default'.

To test this I uninstalled both ngsolve and netgen-mesher and reinstalled both using the command
pip3 install ngsolve==6.2.2204

Paul Ledger - 2025 
Added new boundary layer capability
"""



# Setting mur, sigma, alpha, and defining the top level object name:
material_name = ['mat1']
mur = [100]
sigma = [1e6]
alpha = 0.001

# Boundary Layer Settings: max frequency under consideration, the total number of prismatic layers and the material of each layer.
# Setting Boundary layer Options:
max_target_frequency = 1e8
boundary_layer_material = material_name[0]
number_of_layers = 2


# setting half hox side lengths
s = 1

# Generating OCC primative cube centered at [0,0,0] with side lengths 2*s:
cube = Box(Pnt(-s, -s, -s), Pnt(s,s,s))
print(cube.mass)

# setting material and bc names:
# For compatability, we want the non-conducting region to have the 'outer' boundary condition and be labeled as 'air'
cube.bc('default')
cube.mat(material_name[0])
cube.maxh = 0.2

# Generating a large non-conducting region. For compatability with MPT-Calculator, we set the boundary condition to 'outer'
# and the material name to 'air'.
box = Box(Pnt(-1000, -1000, -1000), Pnt(1000,1000,1000))
box.mat('air')
box.bc('outer')
box.maxh=1000
box=box-cube

# Joining the two meshes:
# Glue joins two OCC objects together without interior elemements
joined_object = Glue([cube, box])

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
nmesh = OCCGeometry(joined_object).GenerateMesh(boundary_layers=[B]) 

nmesh.Save(r'VolFiles/OCC_box_prism_100.vol')
# print(nmesh.GetMaterial(2))
from ngsolve import *
mesh = Mesh(nmesh)
print(f'Materials = {mesh.GetMaterials()}')
