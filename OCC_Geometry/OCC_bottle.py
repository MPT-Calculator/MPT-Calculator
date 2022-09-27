from netgen.occ import *
import math

myHeight = 70
myWidth = 50
myThickness = 30

pnt1 = Pnt(-myWidth / 2., 0, 0)
pnt2 = Pnt(-myWidth / 2., -myThickness / 4., 0)
pnt3 = Pnt(0, -myThickness / 2., 0)
pnt4 = Pnt(myWidth / 2., -myThickness / 4., 0)
pnt5 = Pnt(myWidth / 2., 0, 0)

seg1 = Segment(pnt1, pnt2)
arc = ArcOfCircle(pnt2, pnt3, pnt4)
seg2 = Segment(pnt4, pnt5)

wire = Wire([seg1, arc, seg2])
mirrored_wire = wire.Mirror(Axis((0, 0, 0), X))
w = Wire([wire, mirrored_wire])

f = Face(w)
body = f.Extrude(myHeight * Z)
body = body.MakeFillet(body.edges, myThickness / 12.)

neckax = Axes(body.faces.Max(Z).center, Z)
myNeckRadius = myThickness / 4.
myNeckHeight = myHeight / 10
neck = Cylinder(neckax, myNeckRadius, myNeckHeight);
body = body + neck

fmax = body.faces.Max(Z)
thickbody = body.MakeThickSolid([fmax], -myThickness / 50, 1.e-3)

cyl1 = Cylinder(neckax, myNeckRadius * 0.99, 1).faces[0]
cyl2 = Cylinder(neckax, myNeckRadius * 1.05, 1).faces[0]
aPnt = Pnt(2. * math.pi, myNeckHeight / 2.)
aDir = Dir(2. * math.pi, myNeckHeight / 4.)
anAx2d = gp_Ax2d(aPnt, aDir)
aMajor = 2. * math.pi
aMinor = myNeckHeight / 10
arc1 = Ellipse(anAx2d, aMajor, aMinor).Trim(0, math.pi)
arc2 = Ellipse(anAx2d, aMajor, aMinor / 4).Trim(0, math.pi)
seg = Segment(arc1.start, arc1.end)

wire1 = Wire([Edge(arc1, cyl1), Edge(seg, cyl1)])
wire2 = Wire([Edge(arc2, cyl2), Edge(seg, cyl2)])
threading = ThruSections([wire1, wire2])

res = thickbody+threading

geo = OCCGeometry(res)
nmesh = geo.GenerateMesh()
nmesh.Save(r'VolFiles/OCC_bottle.vol')
