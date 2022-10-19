algebraic3d

solid rest = orthobrick (-1000,-1000,-1000; 1000,1000,1000);
solid brick1 = orthobrick (-5,-5,-5; 5,5,5) -maxh=1;

solid domain = rest and not brick1 -maxh=1000;

tlo domain -transparent -col=[0,0,1];#air
tlo brick1 -col=[1,0,0];#cube -mur=1 -sig=1E+06
