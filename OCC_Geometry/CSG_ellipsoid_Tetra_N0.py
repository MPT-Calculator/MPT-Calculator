from netgen.csg import *
from Functions.Helper_Functions.Add_material import *

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
material_name = ['ell']
subdomain_dict = {'ell':material_name[0]}

mur = [32]
sigma = [1e7]
alpha = 0.001

# Boundary Layer Settings: max frequency under consideration, the total number of prismatic layers and the material of each layer.
# Setting Boundary layer Options:
max_target_frequency = 1e8
boundary_layer_material = material_name[0]
number_of_layers = 2


# Update the geometry file to contain -material flags to allow mesh with boundary layers to be generated
src = r'GeoFiles/Ellipsoid_N0.geo'
Add_material(src,subdomain_dict)
geo = CSGeometry(r'GeoFiles/Ellipsoid_N0_append.geo')


#neg_ellipsoid = ellipsoid - Box(Pnt(-100,-100,-100), Pnt(0,100,100))
#pos_ellipsoid = ellipsoid - Box(Pnt(0,-100,-100), Pnt(100,100,100))
#ellipsoid = pos_ellipsoid + neg_ellipsoid

# setting material and bc names:
# For compatability, we want the non-conducting region to have the 'outer' boundary condition and be labeled as 'air'
#ellipsoid.bc('default')
#ellipsoid.mat(material_name[0])
#ellipsoid.maxh = 0.15

# Generating a large non-conducting region. For compatability with MPT-Calculator, we set the boundary condition to 'outer'
# and the material name to 'air'.
#box = Box(Pnt(-1000, -1000, -1000), Pnt(1000,1000,1000))
#box.mat('air')
#box.bc('outer')
#box.maxh=1000
#box=box-ellipsoid

# Joining the two meshes:
# Glue joins two OCC objects together without interior elemements
#joined_object = Glue([ellipsoid, box])

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
nmesh = geo.GenerateMesh(meshsize.coarse,boundary_layers=[B]) 

# Setting boundary condition name for outer boundary
# This is not possible in the geo file that only allow integer indexes.
# We chose the first subdomain to be a box and so that means the first 6 boundaries are the outer ones.
for i in range(6):
    nmesh.SetBCName(i, 'outer')

nmesh.Save(r'VolFiles/CSG_ellipsoid_Tetra_N0.vol')
# print(nmesh.GetMaterial(2))
from ngsolve import *
mesh = Mesh(nmesh)
# High curvature so use low curve when running MPT-Calculator.
print(f'Materials = {mesh.GetMaterials()}')
