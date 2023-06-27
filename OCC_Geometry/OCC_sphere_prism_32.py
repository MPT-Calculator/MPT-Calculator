from netgen.occ import *
# from ngsolve import *

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


# setting radius
r = 1

# Generating OCC primative sphere centered at [0,0,0] with radius r:
sphere = Sphere(Pnt(0,0,0), r=r)

pos_sphere = sphere - Box(Pnt(0,100,100), Pnt(-100,-100,-100))
neg_sphere = sphere - Box(Pnt(0,100,100), Pnt(100,-100,-100))
sphere = pos_sphere + neg_sphere

# setting material and bc names:
# For compatability, we want the non-conducting region to have the 'outer' boundary condition and be labeled as 'air'
sphere.bc('default')
sphere.mat(material_name[0])
sphere.maxh = 0.2

# Generating a large non-conducting region. For compatability with MPT-Calculator, we set the boundary condition to 'outer'
# and the material name to 'air'.
box = Box(Pnt(-1000, -1000, -1000), Pnt(1000,1000,1000))
box.mat('air')
box.bc('outer')
box.maxh=1000

# Joining the two meshes:
# Glue joins two OCC objects together without interior elemements
joined_object = Glue([sphere, box])

# Generating Mesh:
nmesh = OCCGeometry(joined_object).GenerateMesh()


# Creating Boundary Layer Structure:
mu0 = 4 * 3.14159 * 1e-7
tau = (2/(max_target_frequency * sigma[0] * mu0 * mur[0]))**0.5 / alpha
layer_thicknesses = [(2**n)*tau for n in range(number_of_layers)]

nmesh.BoundaryLayer(boundary=".*", thickness=layer_thicknesses, material=boundary_layer_material,
                           domains=boundary_layer_material, outside=False)


nmesh.Save(r'VolFiles/OCC_sphere_prism_32.vol')
# print(nmesh.GetMaterial(2))
from ngsolve import *
mesh = Mesh(nmesh)
print(f'Materials = {mesh.GetMaterials()}')
