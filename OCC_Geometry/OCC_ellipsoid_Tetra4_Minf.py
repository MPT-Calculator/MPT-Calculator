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
mur = [32]
sigma = [1e7]
alpha = 0.01

# Boundary Layer Settings: max frequency under consideration, the total number of prismatic layers and the material of each layer.
# Setting Boundary layer Options:
max_target_frequency = 1e8
boundary_layer_material = material_name[0]
number_of_layers = 2


# setting radius
r1 = 5.02646033
r2 = 3.14127911
r3 = 0.01624702


# Generating OCC primative sphere centered at [0,0,0] with radius r:
#ellipsoid = Ellipsoid(Axes(Pnt(0,0,0),n=Z,h=X),r1,r2,r3)
# Follow this tutorial to get an ellipsoid
#https://forum.ngsolve.org/t/drawing-mesh-and-generating-ellipsoid/2213/2
sp = Sphere( (0,0,0), 1)
gtr = gp_GTrsf( (r1,0,0, 0,r2,0, 0,0,r3), (0,0,0))
ellipsoid = gtr (sp)

#neg_ellipsoid = ellipsoid - Box(Pnt(-100,-100,-100), Pnt(0,100,100))
#pos_ellipsoid = ellipsoid - Box(Pnt(0,-100,-100), Pnt(100,100,100))
#ellipsoid = pos_ellipsoid + neg_ellipsoid

#ellipsoid = Ellipsoid((0, 0, 0), (r1, 0, 0), (0, r2, 0), (0, 0, r3))

# setting material and bc names:
# For compatability, we want the non-conducting region to have the 'outer' boundary condition and be labeled as 'air'
ellipsoid.bc('default')
ellipsoid.mat(material_name[0])
ellipsoid.maxh = 1.0


# Slice the ellipsoid with a HalfSpace along the XY-plane (Z=0)
# This splits the single closed shell into a distinct Top and Bottom face
cutting_plane = HalfSpace((0,0,0), (0,0,1)) # Facing +Z direction
top_half = ellipsoid * cutting_plane
bot_half = ellipsoid - cutting_plane

# Glue them back into a single volumetric solid with an internal seam
ellipsoid_sliced = Glue([top_half, bot_half])

# 3. Target the top and bottom curved exterior faces
# We select them based on their center of gravity in the Z direction
top_face = ellipsoid_sliced.faces.Max(Z)
bot_face = ellipsoid_sliced.faces.Min(Z)

# 4. Apply the CloseSurfaces Identification 
# This tells Netgen: Copy the mesh layout from top_face down to bot_face
top_face.Identify(bot_face, name="ellipsoid_close", type=IdentificationType.CLOSESURFACES)


# Generating a large non-conducting region. For compatability with MPT-Calculator, we set the boundary condition to 'outer'
# and the material name to 'air'.
box = Box(Pnt(-1000, -1000, -1000), Pnt(1000,1000,1000))
box.mat('air')
box.bc('outer')
box.maxh=1000
box=box-ellipsoid_sliced

# Joining the two meshes:
# Glue joins two OCC objects together without interior elemements
joined_object = Glue([ellipsoid_sliced, box])

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
nmesh = OCCGeometry(joined_object).GenerateMesh(meshsize.coarse,curvaturesafety=6.0,segmentsperedge=100) 
#boundary_layers=[B]
nmesh.Save(r'VolFiles/OCC_ellipsoid_Tetra4_Minf.vol')
# print(nmesh.GetMaterial(2))
from ngsolve import *
mesh = Mesh(nmesh)
# High curvature so use low curve when running MPT-Calculator.
print(f'Materials = {mesh.GetMaterials()}')
