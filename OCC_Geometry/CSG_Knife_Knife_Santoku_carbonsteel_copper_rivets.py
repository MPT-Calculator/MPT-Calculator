from Functions.Helper_Functions.Add_material import *
from netgen.csg import *
from netgen.meshing import BoundaryLayerParameters

from netgen.occ import *

"""
Example of reading a CSG geometry defined in a .geo file (complex geometry with multiple components)y.

Paul Ledger - 2025
Added new boundary layer capability
Included fix to allow CSG geometries to be used by automatically inserting "-material=.."
"""
# List of materials
material_name = ['carbon_steel', 'copper']
# Dictionary mapping subdomains to materials
subdomain_dict = {'Blade':material_name[0],"Bolster":material_name[0],"Tang":material_name[0],"Rivets":material_name[1]}
sigma = [1.6e6, 58e6]
mur = [100, 1]
alpha = 0.001

# Setting Boundary layer Options:
max_target_frequency = 1e10
boundary_layer_material = material_name[0]
number_of_layers = 3

# Update the geometry file to contain -material flags to allow mesh with boundary layers to be generated
src=r'GeoFiles/Knife_Santoku.geo'
Add_material(src,subdomain_dict)
geo = CSGeometry(r'GeoFiles/Knife_Santoku_append.geo')

# Applying Boundary Layers:
mu0 = 4 * 3.14159 * 1e-7
tau = (2/(max_target_frequency * sigma[0] * mu0 * mur[0]))**0.5 / alpha
layer_thicknesses = [(2**n)*tau for n in range(number_of_layers)]

B = BoundaryLayerParameters(boundary=".*", thickness=layer_thicknesses, new_material=boundary_layer_material,
                           domain=boundary_layer_material, outside=False, disable_curving=False )
nmesh = geo.GenerateMesh(boundary_layers=[B])

# Add boundary conditions
nmesh.SetBCName(0, 'outer')
nmesh.SetBCName(1, 'outer')
nmesh.SetBCName(2, 'outer')
nmesh.SetBCName(3, 'outer')
nmesh.SetBCName(4, 'outer')
nmesh.SetBCName(5, 'outer')

nmesh.Save(r'VolFiles/CSG_Knife_Knife_Santoku_carbonsteel_copper_rivets.vol')
