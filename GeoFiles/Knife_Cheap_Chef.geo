algebraic3d


solid Domain = orthobrick(-1000,-1000,-1000;1000,1000,1000);


#Create the blade
solid blade = ellipticcylinder(0,0,0;200,0,0;0,31.5,0)
    and ellipticcylinder(0,-13.0,0;222.0,0,0;0,31.5,0)
    and plane(0,0,0;-1,0,0)
    and plane(0,18.5,-0.75;0,0,-1)
    and plane(0,18.5,0.75;0,0,1)-maxh=10;



#Create the bolster
solid heelblank = orthobrick(-30,-31.5,-0.875;0,18.5,0.875);

solid heelcut1 = plane(0,-1.5,0;0,1,0)
    and plane(-15.0,0,0;1,0,0);

solid heelcut2 = plane(0,-16.5,0;0,1,0);

solid heelcut3 = cylinder(-14.7,-16.3,-1;-14.7,-16.3,1;15.0);

solid heelcut = heelcut1 or heelcut2 or heelcut3;

solid heeltaper = plane(0,0,0.75;-0.125,0,-30)
    or plane(0,0,-0.75;-0.125,0,30);
solid heel = heelblank and not heelcut and not heeltaper-maxh=10;


#Create the tang
solid tang = orthobrick(-110,-1.5,-0.875;-30,18.5,0.875)
    and plane(-30,18.5,0;-5.0,80,0)
    and plane(-30,-1.5,0;-5.0,-80,0)-maxh=5;




solid knife = blade or heel or tang;
solid rest = Domain and not knife;

tlo rest -transparent -col=[0,0,1];#air
tlo knife -col=[1,0,0];#Knife -mur=5 -sig=1.6E+06


