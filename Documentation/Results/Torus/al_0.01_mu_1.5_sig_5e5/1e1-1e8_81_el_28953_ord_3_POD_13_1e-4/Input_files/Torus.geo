algebraic3d

solid sphout = sphere (0, 0, 0; 100);
solid torin = torus (0, 0, 0; 1,0,0;2; 1) -maxh=0.4;

solid rest = sphout and not torin;

tlo rest -transparent -col=[0,0,1];#air
tlo torin -col=[1,0,0];#torus -mur=1.5 -sig=5E+05
