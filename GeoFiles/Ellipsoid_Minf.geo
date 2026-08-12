algebraic3d
#
# Example with two sub-domains: 
#
solid box = orthobrick (-100, -100, -100; 100, 100, 100);
#solid ellpout = ellipsoid (0, 0, 0; 100, 0, 0; 0, 100, 0; 0, 0, 200);
# Important the name below must match the material name in the CSG.py file
solid ell = ellipsoid (0, 0, 0; 2.93348328, 0, 0; 0, 1.90849167, 0; 0, 0, 1.3700185) -maxh=0.2;

solid object=ell;
solid rest = box and not object;

tlo rest -transparent -col=[0,0,1] ;#air
tlo object -col=[1,0,0] ;#ell -mur=32 -sig=1E+07

