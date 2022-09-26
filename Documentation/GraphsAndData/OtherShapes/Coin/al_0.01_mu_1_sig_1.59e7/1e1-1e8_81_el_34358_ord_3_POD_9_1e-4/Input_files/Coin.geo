algebraic3d

solid boxout = orthobrick (-100, -100, -100; 100, 100, 100);
solid cylin = cylinder ( 0, 0, -0.1575; 0, 0, 0.1575; 1.125 )
	and plane (0, 0, -0.1575; 0, 0, -1)
	and plane (0, 0, 0.1575; 0, 0, 1) -maxh=0.1;

solid rest = boxout and not cylin;

tlo rest -transparent -col=[0,0,1];#air
tlo cylin-col=[1,0,0];#coin -mur=1 -sig=1.59E+07
