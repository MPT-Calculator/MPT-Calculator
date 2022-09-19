algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);
solid cube1 = orthobrick(0,  -10,-10;20,10,10) -maxh=2;
solid cube2 = orthobrick(-20,-10,-10; 0,10,10) -maxh=2;

solid rest = boxout and not (cube1 and cube2);


tlo rest -transparent -col=[0,0,1];#air
tlo cube1 -col=[1,0,0];#cube1 -mur=1 -sig=5.8e7
tlo cube2 -col=[0,1,0];#cube2 -mur=1 -sig=5.8e7
