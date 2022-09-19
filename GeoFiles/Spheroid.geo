algebraic3d
#
# Example with two sub-domains: 
#
solid ellpout = ellipsoid (0, 0, 0; 100, 0, 0; 0, 100, 0; 0, 0, 200);
solid ellpin = ellipsoid (0, 0, 0; 1, 0, 0; 0, 1, 0; 0, 0, 2) -maxh=0.1;

solid rest = ellpout and not ellpin;

tlo rest -transparent -col=[0,0,1];#air
tlo ellpin -col=[1,0,0];#spheroid -mur=1.5 -sig=1E+06