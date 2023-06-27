algebraic3d


solid sphout = sphere (0, 0, 0; 100);
solid sphin = sphere (0, 0, 0; 1) -maxh=0.08;

solid rest = sphout and not sphin;

tlo rest -transparent -col=[0,0,1];#air
tlo sphin -col=[1,0,0];#sphere -mur=10 -sig=1e6
