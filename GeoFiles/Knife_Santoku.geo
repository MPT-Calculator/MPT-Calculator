algebraic3d


solid Domain = orthobrick(-1000,-1000,-1000;1000,1000,1000);


#Create the blade
solid bladeblank = orthobrick(0,-27.8,-0.637;170,16.4,0.637)
    and plane(0,16.4,-0.637;0,-0.51,-44.2)
    and plane(0,16.4,0.637;0,-0.51,44.2);

solid bladeFrontCut = orthobrick(126.0,-28.3,-1.27;172.0,16.8,1.27)
    and not cylinder(126.0,-27.8,-1.27;126.0,-27.8,1.27;44.4);

solid blade = bladeblank and not bladeFrontCut -maxh=10;



#Create the bolster
solid heelmain = orthobrick(-30,-27.8,-8.51;0,16.4,8.51);
solid cutfrontleft = cylinder(0,16.4,-17.0;0,-27.8,-16.5;17.2)
    and plane(0,16.4,-0.637;0,0.51,44.2);
solid cutfrontright = cylinder(0,16.4,17.0;0,-27.8,16.5;17.2)
    and plane(0,16.4,0.637;0,0.51,-44.2);
solid cutbackleft = cylinder(-30,16.4,-8.51;-30,-27.8,-8.51;6.76);
solid cutbackright = cylinder(-30,16.4,8.51;-30,-27.8,8.51;6.76);
solid cutunder1 = cylinder(-27.0,-25.631624994156027,-8.51;-27.0,-25.631624994156027,8.51;12.4);
solid cutunder2 = plane(-15.0,-25.631624994156027,0;1,0,0)
    and plane(-15.0,-25.631624994156027,0;0,1,0);
solid cutunder3 = plane(-27.0,-13.6,0;0,1,0)
    and plane(-27.0,-13.6,0;1,0,0);
solid heelrounder = ellipticcylinder(0,-3.05,0;0,-25.631624994156027,0;0,0,7.74);

solid cutunder = cutunder1 or cutunder2 or cutunder3;
solid heelcuts = cutfrontleft or cutfrontright or cutbackleft or cutbackright or cutunder1 or cutunder;

solid heel = heelmain and heelrounder and not heelcuts and not heelcuts-maxh=20;



#Create the tang
solid tangblank = orthobrick(-150,-13.6,-1.75;-30,16.4,1.75)
    and ellipticcylinder(-90.0,16.4,0;84.0,0,0;0,24.0,0);

solid tangfront = orthobrick(-56.4,-13.6,-1.75;-30,16.4,1.75)
    and not cylinder(-54.0,-16.6,-1;-54.0,-16.6,1;11.4);

solid tangback = orthobrick(-150,-13.6,-1.75;-124.0,16.4,1.75)
    and not cylinder(-126.0,-16.6,-1;-126.0,-16.6,1;11.4);

solid tangend = orthobrick(-132.0,-13.6,-1.75;-30,16.4,1.75)
    or cylinder(-132.0,1.4,-1;-132.0,1.4,1;15.3);



#Add the rivets
# For prisms we bisect the rivets.
solid topbox = orthobrick(-1000, 4.4, -1000; 1000, 1000, 1000);

solid rivet1 = cylinder(-47.5,4.4,-8.0;-47.5,4.4,8.0;3.5)
    and plane(0,0,-10.0;0,0,-1)
    and plane(0,0,10.0;0,0,1);

solid tophalf1 = rivet1 and not topbox;
solid bottomhalf1 = rivet1 and topbox;
solid rivet1split = tophalf1 or bottomhalf1;

solid rivet2 = cylinder(-90.0,4.4,-8.0;-90.0,4.4,8.0;3.5)
    and plane(0,0,-10.0;0,0,-1)
    and plane(0,0,10.0;0,0,1);

solid tophalf2 = rivet2 and not topbox;
solid bottomhalf2 = rivet2 and topbox;
solid rivet2split = tophalf2 or bottomhalf2;

solid rivet3 = cylinder(-132.0,4.4,-8.0;-132.0,4.4,8.0;3.5)
    and plane(0,0,-10.0;0,0,-1)
    and plane(0,0,10.0;0,0,1);

solid tophalf3 = rivet3 and not topbox;
solid bottomhalf3 = rivet3 and topbox;
solid rivet3split = tophalf3 or bottomhalf3;

solid rivets = rivet1split or rivet2split or rivet3split -maxh=1;

solid tangblock = tangblank or tangfront or tangback;
solid tang = tangblock and tangend and not rivets;




solid rest = Domain and not blade and not heel and not tang and not rivets -maxh=1000;

tlo rest -transparent -col=[0,0,1];#air
tlo blade -col=[1,0,0];#Blade -mur=5 -sig=1.6E+06
tlo heel -col=[0,1,0];#Bolster -mur=5 -sig=1.6E+06
tlo tang -col=[0,0,1];#Tang -mur=5 -sig=1.6E+06
tlo rivets -col=[1,1,0];#Rivets -mur=1 -sig=5.8E+07

