from netgen.occ import *
from netgen.meshing import BoundaryLayerParameters
import numpy as np
from matplotlib import pyplot as plt

# Setting mur, sigma, alpha, and defining the top level object name:
material_name = ['mat1']
mur = [1]
sigma = [1e6]
alpha = 0.01

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
sphere.bc('default')
sphere.mat(material_name[0])
sphere.maxh = 0.2

# Generating a large non-conducting region. 
box = Box(Pnt(-1000, -1000, -1000), Pnt(1000,1000,1000))
box.mat('air')
box.bc('outer')
box.maxh=1000
box=box-sphere

# Joining the two meshes:
# Glue joins two OCC objects together without interior elemements
joined_object = Glue([sphere, box])

# Creating Boundary Layer Structure:
mu0 = 4 * 3.14159 * 1e-7
tau = (2/(max_target_frequency * sigma[0] * mu0 * mur[0]))**0.5 / alpha
layer_thicknesses = [(2**n)*tau for n in range(number_of_layers)]

B = BoundaryLayerParameters(boundary=".*", thickness=layer_thicknesses, new_material=boundary_layer_material,
                           domain=boundary_layer_material, outside=False, disable_curving=False)

nmesh = OCCGeometry(joined_object).GenerateMesh(boundary_layers=[B])

print("Mesh generated")

from ngsolve import *
mesh = Mesh(nmesh)
print(f'Materials = {mesh.GetMaterials()}')

import warnings
#import osii
import gc

# Testing of convergence of integration of bilinear fomrs

# Setting mesh curve order so that we can recheck for linear geometry:
curve_order=5
mesh.Curve(curve_order)
print('Mesh curve order set to: ', curve_order)

# Checking K:
# defining finite element space.
dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
order=3
fes2 = HCurl(mesh, order=3, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    
# Creating the bilinear form
u, v = fes2.TnT()
rel_diff = 1
counter = 1
rel_diff_array = []
ord_array = []
bonus_intord = 2

bilinearform_tol= 1e-10
max_iter=10
drop_tol=0

while (rel_diff > bilinearform_tol) and (counter < max_iter):
    print(f'K: Iteration {counter}: bonus_intord = {bonus_intord}')
    K = BilinearForm(fes2, symmetric=True, delete_zero_elements =drop_tol, keep_internal=False, symmetric_storage=True)
    K += SymbolicBFI(curl(u) * (curl(v)), bonus_intorder=bonus_intord)

    with TaskManager():
        K.Assemble()

        
    if counter == 1:  # first iteration
        nvalsold = np.linalg.norm(K.mat.AsVector()[:])
    else:
        last_rel_diff = rel_diff
        nvals = np.linalg.norm(K.mat.AsVector()[:])
        rel_diff = np.abs(nvals-nvalsold)/nvals
        nvalsold = nvals
        rel_diff_array += [rel_diff]
        ord_array += [bonus_intord]
        del K
        
        # using bonus_intord =n and bonus_intord =n+1 gives the same result (?).
 
    bonus_intord += 2
    counter += 1

"""if counter>=max_iter:
    index=np.argmin(np.array(rel_diff_array))
    bonus_intord=ord_array[index]
    print(index,bonus_intord)
    counter=1"""

plt.figure()
plt.title(f'Curve order: {curve_order}')
plt.plot(ord_array, rel_diff_array, '*-', label='Relative Difference K')
plt.axhline(bilinearform_tol, color='r', label='Tolerance')
plt.xlabel('Integration Order')
plt.ylabel('Relative Difference K')
plt.yscale('log')
plt.legend()
plt.savefig('K_Convergence.pdf')



rel_diff = 1
counter = 1
rel_diff_array = []
ord_array = []
bonus_intord = 2

while (rel_diff > bilinearform_tol) and (counter < max_iter):
    print(f'M: Iteration {counter}: bonus_intord = {bonus_intord}')
    M = BilinearForm(fes2, symmetric=True, delete_zero_elements =drop_tol, keep_internal=False, symmetric_storage=True)
    M += SymbolicBFI(u*v, bonus_intorder=bonus_intord)
        
    with TaskManager():
        M.Assemble()
        
        
    if counter == 1:  # first iteration
        nvalsold = np.linalg.norm(M.mat.AsVector()[:])
    else:
        last_rel_diff = rel_diff
        nvals = np.linalg.norm(M.mat.AsVector()[:])
        rel_diff = np.abs(nvals-nvalsold)/nvals
        nvalsold = nvals
        rel_diff_array += [rel_diff]
        ord_array += [bonus_intord]
        del M
        
        # using bonus_intord =n and bonus_intord =n+1 gives the same result (?).
 
    bonus_intord += 2
    counter += 1

"""if counter>=max_iter:
    index=np.argmin(np.array(rel_diff_array))
    bonus_intord=ord_array[index]
    print(index,bonus_intord)
    counter=1"""

plt.figure()
plt.title(f'Curve order: {curve_order}')
plt.plot(ord_array, rel_diff_array, '*-', label='Relative Difference M')
plt.axhline(bilinearform_tol, color='r', label='Tolerance')
plt.xlabel('Integration Order')
plt.ylabel('Relative Difference M')
plt.yscale('log')
plt.legend()
plt.savefig('M_Convergence.pdf')

