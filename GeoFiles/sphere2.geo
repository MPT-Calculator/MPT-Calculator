algebraic3d


solid sphout = orthobrick (-1000,-1000,-1000; 1000, 1000, 1000);
solid sphin = sphere (0, 0, 0; 1) -maxh=0.2;

solid rest = sphout and not sphin -maxh=1000;

tlo rest -transparent -col=[0,0,1];#air
tlo sphin -col=[1,0,0];#sphere -mur=32 -sig=1e6