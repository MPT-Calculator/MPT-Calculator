algebraic3d


solid Domain = orthobrick(-1000,-1000,-1000;1000,1000,1000);


#Create the blade
solid blade = orthobrick(0,-11.8,-0.625;125,6.94,0.625)
    and ellipticcylinder(0,11.6,0;124.0,0,0;0,24.4,0)-maxh=5;



#Create the bolster
solid heel = orthobrick(-20,-11.8,-0.625;0,6.94,0.625)-maxh=10;



#Create the tang
solid tang = orthobrick(-70,-8.56,-0.625;-20,3.94,0.625)
    and plane(-20,3.94,0;-3.12,50,0)
    and plane(-20,-8.56,0;-3.12,-50,0)-maxh=5;





solid rest = Domain and not blade and not heel and not tang;

tlo rest -transparent -col=[0,0,1];#air
tlo blade -col=[1,0,0];#Blade -mur=5 -sig=1.6E+06
tlo heel -col=[0,1,0];#Bolster -mur=5 -sig=1.6E+06
tlo tang -col=[0,0,1];#Tang -mur=5 -sig=1.6E+06

