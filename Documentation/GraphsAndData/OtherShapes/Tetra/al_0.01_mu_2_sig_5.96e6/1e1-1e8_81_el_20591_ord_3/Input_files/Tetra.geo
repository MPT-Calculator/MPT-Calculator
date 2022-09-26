algebraic3d

solid box = orthobrick (-100, -100, -100; 100, 100, 100);

solid tetra = polyhedron (0.000,0.000,0.000; 7.00,0.000,0.000; 5.5,4.6,0.000; 3.3,2.0,5.0 ;;
                           1,3,2 ; 1,4,3; 1,2,4 ; 2,3,4 ) -maxh=0.3;

solid object= tetra;
solid outside=box and not object ;

tlo outside -col=[0,0,1] -transparent;#air
tlo object -col=[1,0,0] ;#tetra -mur=2 -sig=5.96E+06
