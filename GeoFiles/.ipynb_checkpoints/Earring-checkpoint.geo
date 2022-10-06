algebraic3d


#Outer domain

solid outer_box = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);



#Object parts

solid ball = sphere(6,0,0;1);
solid bar = cylinder(0,0,0;10,0,0;0.5)
	and plane(0,0,0;-1,0,0)
	and plane(6,0,0;1,0,0);
solid end = cylinder(-1,0,0;1,0,0;1.2)
	and plane(0,0,0;1,0,0)
	and plane(-0.5,0,0;-1,0,0);



#Combining parts

solid earing = ball or bar or end -maxh=0.25;
solid rest = outer_box and not earing;



#Defining top level objects

tlo rest -transparent -col=[0,0,1];#air
tlo earing -col=[1,0,0];#earring -mur=1 -sig=4.1E+07
