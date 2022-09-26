algebraic3d

solid rest = sphere (0, 0, 0; 100);
solid brick1 = orthobrick (-1,0,0;0,1,1) -maxh=0.12;
solid brick2 = orthobrick (0,0,0;1,1,1) -maxh=0.12;

solid domain = rest and not brick1 and not brick2;

tlo domain -transparent -col=[0,0,1];#air
tlo brick1 -col=[1,0,0];#mat1 -mur=1 -sig=1E+06
tlo brick2 -col=[0,1,0];#mat2 -mur=1 -sig=1E+08
