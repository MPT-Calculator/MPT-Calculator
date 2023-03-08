algebraic3d

solid sphout = sphere (0, 0, 0; 1000);

solid shaft = orthobrick(-10, -1, -0.5; 10, 1, 0.5) -maxh=0.5;

solid claw = cylinder(11, 0, -0.5; 11, 0, 0.5; 2)
	and plane(0, 0, 0.5; 0, 0, 1)
	and plane(0, -0, -0.5; 0, 0, -1);

solid cutout_claw = orthobrick (10.3, -0.75, -0.51; 13, 0.75, 0.51);

solid socket = cylinder(-11, 0, -1; -11, 0, 1; 2)
	and plane(0, 0, 0.5; 0, 0, 1)
	and plane(0, -0, -0.5; 0, 0, -1);

solid cutout_socket = cylinder(-11.2, 0, -1.1; -11.2, 0, 1.1; 1.3)
	and plane(0, 0, 0.51; 0, 0, 1)
	and plane(0, -0, -0.51; 0, 0, -1);

solid wrench = shaft or claw and not cutout_claw or socket and not cutout_socket;


solid rest = sphout and not wrench;

tlo rest -transparent -col=[0,0,1];#air
tlo wrench -col=[1,0,0];#wrench -mur=1  -sig=3.5E7
