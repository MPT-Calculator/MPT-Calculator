from netgen.occ import *
import numpy as np

# HEre I have chosen to model the bomblet as an aluminium sphere with additional steel flutes.
# The conductivity for aluminium can vary significantly (see https://www.nde-ed.org/NDETechniques/EddyCurrent/ET_Tables/ET_matlprop_Aluminum.xhtml).
# I've chosen a value of 3e7 to be roughly middle of the stated range. Simularly with stainless steel (https://www.nde-ed.org/NDETechniques/EddyCurrent/ET_Tables/ET_matlprop_Iron-Based.xhtml)
# For steel I've chosen 6e6 S/m, which is what we used in the coin and boundary layer paper.
material_name = ['bomblet', 'ring', 'air']
mur = [1, 200, 1]
sigma = [3e7, 6e6, 0]
alpha = 1e-2

# Boundary Layer Settings: max frequency under consideration, the total number of prismatic layers and the material of each layer.
# Setting Boundary layer Options:
max_target_frequency = 1e7
boundary_layer_material = material_name[0]
boundary_layer_domain = material_name[0]
number_of_layers = 2

filename = r'StepFiles/bomblet_uniform_meshlab.step'
# filename = r'../StepFiles\45mm Cylinder.step'


geo = OCCGeometry(filename)
# geo.Heal(tolerance=1e-6, splitpartitions=True)


bomblet = geo.shape.Move((-geo.shape.center.x, -geo.shape.center.y, -geo.shape.center.z))
# bomblet = bomblet.Scale((0,0,0), 0.1)

vol = bomblet.mass
# radius = (3/(4*np.pi) * (vol*1.01))**(1/3)

radius = (bomblet.bounding_box[1][2] - bomblet.bounding_box[0][2])/2
radius_inner = radius - 7.4
radius_cyl = (bomblet.bounding_box[1][0] - bomblet.bounding_box[0][0])/2
h_cyl = 3.11 #(vol - (4*np.pi*radius**3 / 3)) / (np.pi * (radius_cyl**2 - radius**2))

sph_inner = Sphere(Pnt(0,0,0), r=radius_inner)

sphere = Sphere(Pnt(0,0,0), r=radius) - sph_inner
pos_sphere = sphere - Box(Pnt(0,100,100), Pnt(-100,-100,-100))
neg_sphere = sphere - Box(Pnt(0,100,100), Pnt(100,-100,-100))
sphere = Glue([pos_sphere, neg_sphere])

clamp = Cylinder(Pnt(0,0,-h_cyl/2), Z, r=radius_cyl, h=h_cyl)# - sph_inner

new_bomblet = Glue([clamp, sphere, sph_inner])
clamp = new_bomblet - sphere - sph_inner

sphere.bc('default')
sphere.mat(material_name[0])
sphere.maxh = 0.5

clamp.bc('ring')
clamp.mat(material_name[1])
clamp.maxh = 0.2
clamp.name = 'ring'
# clamp.col = (0.5,0.5,0.5)

sph_inner.bc('inner')
sph_inner.mat(material_name[2])
sph_inner.maxh = 1000
sph_inner.name = 'air'

print(f'Object Volume:\n '
      f'Total = {new_bomblet.mass}\n '
      f'Sphere = {sphere.mass}\n '
      f'Sphere Inner = {sph_inner.mass}\n '
      f'Clamp = {clamp.mass}\n '
      f'Clamp+Sphere = {sphere.mass + clamp.mass}\n '
      f'Scaled Total = {new_bomblet.mass * (0.1 * alpha)**3}') # this is before we scale the object by a factor 0.1

new_bomblet = Glue([clamp, sphere, sph_inner])
# new_bomblet = sphere
new_bomblet = new_bomblet.Scale((0,0,0), 0.1)

print(f'Object Volume:\n '
      f'Total = {new_bomblet.mass}\n '
      f'Sphere = {new_bomblet.solids[1].mass + new_bomblet.solids[2].mass}\n '
      f'Sphere Inner = {new_bomblet.solids[3].mass}\n '
      f'Clamp = {new_bomblet.solids[0].mass}\n '
      f'Scaled Total = {new_bomblet.mass * alpha**3}')

bounding_box = Box(Pnt(-100, -100, -100), Pnt(100, 100, 100))
bounding_box.mat('air')
bounding_box.name = 'air'
bounding_box.bc('outer')

geo2 = OCCGeometry(Glue([new_bomblet, bounding_box]))
# geo2.Heal()

nmesh = geo2.GenerateMesh()


# Creating Boundary Layer Structure:
mu0 = 4 * 3.14159 * 1e-7

tau = (2/(max_target_frequency * sigma[1] * mu0 * mur[1]))**0.5 / alpha
layer_thicknesses = [(2**n)*tau for n in range(number_of_layers)]

nmesh.BoundaryLayer(boundary=".*", thickness=layer_thicknesses, material='ring',
                           domains='ring', outside=False)


tau = (2/(max_target_frequency * sigma[0] * mu0 * mur[0]))**0.5 / alpha
layer_thicknesses = [(2**n)*tau for n in range(number_of_layers)]

nmesh.BoundaryLayer(boundary=".*", thickness=layer_thicknesses, material=boundary_layer_material,
                           domains='bomblet', outside=False)





nmesh.Save('VolFiles/OCC_bomblet_clamp_realistic_materials_hollow.vol')
