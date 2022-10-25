from netgen.occ import *
from ngsolve import *
from netgen.webgui import Draw as DrawGeo
import numpy as np

sigma = [5.26E+06, 1.63E+07, 1E6]
mur = [1.15, 1, 100]
material_name = ['Nickel_Silver','Brass','Nickel_plating']

r = 23.43/2
r_inner = 7.4
t = 2.8

a = r * np.sin(30 * np.pi/180) / np.sin(75 * np.pi/180)

wp = WorkPlane()
for i in range(12):
    wp.Line(a).Rotate(30)
face = wp.Face()
frame = face.Extrude(t)

cx = frame.center[0]
cy = frame.center[1]
inner_cyln = Cylinder(Pnt(cx,cy,0),Z, r=r_inner, h=t)
frame_outer = frame - inner_cyln

# inner_cyln.name = object_name[1]
# frame_outer.name = object_name[0]

inner_cyln.mat(material_name[1])
frame_outer.mat(material_name[0])

inner_cyln.bc('inner_cyln')
frame_outer.bc('frame_outer')

inner_cyln.maxh = 1
frame_outer.maxh = 1

coin = Glue([inner_cyln, frame_outer])

box = Box(Pnt(-1000, -1000, -1000), Pnt(1000,1000,1000))
box.mat('air')
box.bc('outer')
box.maxh = 1000

joined_object = Glue([coin, box])
geo = OCCGeometry(joined_object)
nmesh = geo.GenerateMesh()

nmesh.BoundaryLayer(boundary=".*", thickness=[5e-5], material=material_name[2],
                           domains=material_name[1], outside=False)

from ngsolve import *
Mesh = Mesh(nmesh)
print(Mesh.GetMaterials())

nmesh.Save(r'VolFiles/OCC_coin.vol')
