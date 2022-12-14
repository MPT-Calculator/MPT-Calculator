algebraic3d
#
# Example with two sub-domains
#
solid boxout = orthobrick (-100, -100, -100; 100, 100, 100);
solid cylin = cylinder ( 10, 0, -1; -10, 0, -1; 20 )
	and plane (0, 0, -1; 0, 0, -1)
	and plane (0, 0, 1; 0, 0, 1) -maxh=0.08;

solid rest = boxout and not cylin;

tlo rest -transparent -col=[0,0,1];#air
tlo cylin-col=[1,0,0];#cylinder -mur=10 -sig=1E+07