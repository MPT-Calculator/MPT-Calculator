#
## A steelgun
#

algebraic3d

solid boxout = orthobrick (-200, -200, -200; 200, 200, 200);

solid steelgun = polyhedron (0,0,0; 5.8,0,0; 5.8,2,0; 2,2,0; 2,7.6,0; 0,7.6,0; 0,2,0; 0,0,1.5; 5.8,0,1.5; 5.8,2,1.5; 2,2,1.5; 2,7.6,1.5; 0,7.6,1.5; 0,2,1.5 ;; 2,1,7,3 ; 4,7,6,5 ; 8,9,10,14 ; 14,11,12,13 ; 1,2,9,8 ; 2,3,10,9 ; 3,4,11,10 ; 4,5,12,11 ; 5,6,13,12 ; 6,1,8,13 ) -maxh=0.3;

solid rest = boxout and not steelgun;

tlo rest -transparent -col=[0,0,1];#air
tlo steelgun -col=[1,0,0];#gun -mur=100 -sig=5.96E+06