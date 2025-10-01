from netgen.csg import *
from Functions.Helper_Functions.Add_material import *

from netgen.occ import *
from netgen.meshing import BoundaryLayerParameters
from ngsolve import *
import shutil

"""
James Elgy - 2024:
Example of loading a CSG geometry (defined in a geo file) into the python interface.

Paul Ledger - 2025
Added new boundary layer capability
Included fix to allow CSG geometries to be used by automatically insert "-material=.."
"""


#material_name = ['tetra']
# List of materials
material_name = ['tetra']
# Dictionary mapping subdomains to materials
subdomain_dict = {'tetra':material_name[0]}
sigma = [1e7]
mur = [32]
alpha = 0.001

# Setting Boundary layer Options:
max_target_frequency = 1e8
boundary_layer_material = material_name[0]
number_of_layers = 3

# Update the geometry file to contain -material flags to allow mesh with boundary layers to be generated
src = r'GeoFiles/Tetra.geo'
Add_material(src,subdomain_dict)
geo = CSGeometry(r'GeoFiles/Tetra_append.geo')

# Applying Boundary Layers:
mu0 = 4 * 3.14159 * 1e-7
tau = (2/(max_target_frequency * sigma[0] * mu0 * mur[0]))**0.5 / alpha
layer_thicknesses = [(2**n)*tau for n in range(number_of_layers)]

B = BoundaryLayerParameters(boundary=".*", thickness=layer_thicknesses, new_material=boundary_layer_material,
                           domain=boundary_layer_material, outside=False, disable_curving=False )
nmesh = geo.GenerateMesh(boundary_layers=[B])

# Setting boundary condition name for outer boundary
for i in range(6):
    nmesh.SetBCName(i, 'outer')

#Old syntax for addeding boundary layers to an existing mesh
#nmesh.BoundaryLayer(boundary=".*", thickness=layer_thicknesses, material=boundary_layer_material, domains=boundary_layer_material, outside=False)
print("mesh generated")

nmesh.Save(r'VolFiles/CSG_Tetra.vol')
