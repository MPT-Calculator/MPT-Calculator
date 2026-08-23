
from netgen.occ import *
from netgen.webgui import Draw as DrawGeo
from netgen.meshing import BoundaryLayerParameters

# Set materials
material_name = ['Steel','StainSteel']
sigma = [4.5e6, 1.45e6]
mur = [100, 1]# reduced mU-r for testing
alpha =0.01

# Boundary layer options
max_target_frequency = 1e8
number_of_layers = 3
boundary_layer_material = material_name[0]

# Reciever
box = Box(Pnt(12,-0.5,-15.75), Pnt(20,0.5,-0.75))
# Barrel
cyl = Cylinder(Pnt(0,0,0), X, r=2, h=20)
cyl_inside = Cylinder(Pnt(0,0,0), X, r=1, h=18)
# Barrel-extra
barrel_ex = Box(Pnt(0,-2.6,-0.5),(20,-1.6,0.5))

#cyl = cyl+ barrel_ex
box2 = Box(Pnt(12,-0.8,-3.75), Pnt(20,0.5,-0.75))

# Reciever is the box without the barrel
#barrel_ex_no_cyl = barrel_ex - cyl - cyl_inside
barrel=cyl-cyl_inside #+ barrel_ex_no_cyl

#barrel = Glue([
#cyl-cyl_inside,
#barrel_ex
#])

rec=box-cyl
rec = rec +box2
#barrel_ex_no_cyl = barrel_ex 
#barrel= Glue([barrel,barrel_ex])

#bs=barrel.shape
#bs=bs.Heal()
#barrel=OCCGeoemtry(bs)
#rs=rec.shape
#rs=rc.Heal()
#rec=OCCGeoemtry(rs)

# Set materials
barrel.mat(material_name[0])
rec.mat(material_name[1])
#barrel_ex_no_cyl.mat(material_name[0])

print("The volume of the object is",barrel.mass+rec.mass)
print("The volume of the barrel is",barrel.mass)
#print("The volume of the barrel ex is",barrel_ex_no_cyl.mass)
print("The volume of the reciever is",rec.mass)
vol=barrel.mass+rec.mass#+barrel_ex_no_cyl.mass
volb=barrel.mass#+barrel_ex_no_cyl.mass
volr=rec.mass
print("The average conductivity is",(sigma[0]*volb+sigma[1]*volr)/vol)
print("The average permeability is",(mur[0]*volb+mur[1]*volr)/vol)


# Set bcs
#barrel.bc("gunbc")
#rec.bc("gunbc")
barrel.bc("default")
rec.bc("default")
#barrel_ex_no_cyl.bc("default")

barrel.maxh = 1.0
rec.maxh = 1.0
# Join objects together to form the gun
gun = Glue([rec,barrel])#,barrel_ex_no_cyl])
DrawGeo (gun)
print("The volume of the gun",gun.mass)

# Add large outer box
box = Box(Pnt(-1000, -1000, -1000), Pnt(1000,1000,1000))
box.mat('air')
box.bc('outer')
box.maxh = 1000


# Join gun to the box
box=box-gun
joined_object = Glue([gun, box])
geo = OCCGeometry(joined_object)
#nmesh = geo.GenerateMesh(grading=0.5)

# Setup the boundary layers
mu0 = 4 * 3.14159 * 1e-7
s = alpha / 1e-3
tau = (2/(max_target_frequency * sigma[0] * mu0 * mur[0]))**0.5 /(alpha/s)
print(s,tau)
layer_thicknesses = [ (2**n)*tau for n in range(number_of_layers)]

#nmesh.BoundaryLayer(boundary=".*", thickness=layer_thicknesses, material=material_name[0],
#                           domains=material_name[0], outside=False)

# New NG-Solve
if number_of_layers> 0:
    #B = BoundaryLayerParameters(boundary=".*", thickness=layer_thicknesses, new_material=boundary_layer_material,domain=boundary_layer_material, outside=False)#, disable_curving=False)

    #nmesh = OCCGeometry(joined_object).GenerateMesh(meshsize.coarse, boundary_layers=[B])
    # Add layers to each conducting subdomain
    layers=[]
    cnt=0
    for mat in [material_name[0],material_name[1]]:
        print(mat)

        tau = (2/(max_target_frequency * sigma[0] * mu0 * mur[0]))**0.5 /(alpha/s)
        layer_thicknesses = [ (2**n)*tau for n in range(number_of_layers)]

        layers.append( BoundaryLayerParameters(boundary=".*", thickness=layer_thicknesses, new_material=mat, domain=mat, outside=False))#, disable_curving=False))
        cnt+=1

    nmesh = OCCGeometry(joined_object).GenerateMesh(meshsize.coarse, boundary_layers=layers)

else:
    nmesh = OCCGeometry(joined_object).GenerateMesh(meshsize.coarse)

#tau = (2/(max_target_frequency * sigma[1] * mu0 * mur[1]))**0.5 /alpha
#layer_thicknesses = [ (2**n)*tau for n in range(number_of_layers)]

#nmesh.BoundaryLayer(boundary=".*", thickness=layer_thicknesses, material=material_name[1],
#                           domains=material_name[1], outside=False)

#Save the mesh
print("save mesh")
nmesh.Save(r'VolFiles/OCC_Gun_modelv3_nonsym_StainSt.vol')

