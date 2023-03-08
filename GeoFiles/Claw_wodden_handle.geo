algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);

solid head_back = ellipticcylinder(0,-20,0;55,0,0;0,40,0)
	and plane(0,0,-12.5;0,0,-1)
	and plane(0,0,12.5;0,0,1)
	and plane(0,0,0;1,0,0)
	and not ellipticcylinder(-10,-35,0;55,0,0;0,40,0)
	and not ellipsoid(-17.5,-30,0;0,0,7.5;-20,45,0;-45,-20,0);

solid head_front = orthobrick(0,-17.5,-12.5;70,20,12.5)
	and (cylinder(47.5,7.5,0;46.5,7.5,0;12)
		or plane(47.5,0,0;1,0,0))
	and not orthobrick(40,-45,-15;80,-5,15)
	and not torus(50,7.5,0;1,0,0;25.5;16.5)
	and not orthobrick(5,-30,-7.5;30,30,7.5);


solid head = head_front or head_back-maxh=5;

solid hammer = head;
solid rest = boxout and not hammer;

tlo rest -transparent -col=[0,0,1];#air
tlo hammer -col=[1,0.25,0.25];#ring -mur=49.99999999999999  -sig=1.7e6