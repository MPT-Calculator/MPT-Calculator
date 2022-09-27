from netgen.occ import *

wp_1 = WorkPlane(Axes((0,0,0), n=Z, h=X))
wp_2 = WorkPlane(Axes((0,1,0), n= -Y, h=X))
wp_3 = WorkPlane(Axes((0,1,0), n= Z, h=X))
wp_4 = WorkPlane(Axes((0,1,0), n= Z, h=X))


for i in range(3):
    wp_1.Line(1).Rotate(120)
face_1 = wp_1.Face()

for i in range(3):
    wp_2.Line(1).Rotate(120)
face_2 = wp_2.Face()

for i in range(3):
    wp_3.Line(1).Rotate(120)
face_3 = wp_3.Face()

for i in range(3):
    wp_4.Line(1).Rotate(120)
face_4 = wp_4.Face()




geo = OCCGeometry(Glue([face_1, face_4, face_3, face_2]))
nmesh = geo.GenerateMesh()
nmesh.Save(r'VolFiles/OCC_plane.vol')