algebraic3d
#
# Example with two sub-domains:  was 10
#
solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


solid cylin1inend = cylinder (0,0,-21.59;0,0,  -21.09; 4.8)        
        and plane (0,0,-21.59; 0,0,-1)
        and plane (0,0,-21.09; 0,0,1);
	
solid cylin1out = cone (0,0,-21.09; 4.8;  0,0,10.54; 4.535)        
        and plane (0,0,-21.09;0,0, -1)
        and plane (0,0,10.54;0,0, 1);
	
solid cone1out = cone (0,0,10.54; 4.535; 0,0,13.65; 3.215)        
        and plane (0,0,10.54; 0,0,-1)
        and plane (0,0,13.65; 0,0,1);

solid cylin2out = cylinder (0,0,13.65; 0,0, 21.59; 3.215)        
        and plane (0,0,13.65; 0,0,-1)
        and plane (0,0,21.59; 0,0,1);



solid cylin1in = cone (0,0,-21.09; 4.3;  0,0,10.54; 4.035)        
        and plane (0,0,-21.09;0,0, -1)
        and plane (0,0,10.54;0,0, 1);


solid cone1in = cone (0,0,10.54; 4.035; 0,0,13.65; 2.715)        
        and plane (0,0,10.54; 0,0,-1)
        and plane (0,0,13.65; 0,0,1);
	
solid cylin2in =  cylinder (0,0,13.65; 0,0, 21.59; 2.715)        
        and plane (0,0,13.65; 0,0,-1)
        and plane (0,0,21.59; 0,0,1);




solid shell1 = cylin1out  and not cylin1in -maxh=0.8;
solid shellend = cylin1inend  -maxh=0.8;

solid shell2 = cone1out  and not cone1in  -maxh=0.8;

solid shell3 = cylin2out  and not cylin2in -maxh=0.8;

solid rest1 = cylin1out and cylin1in;
solid rest2 = cone1out  and cone1in;
solid rest3 = cylin2out  and cylin2in;
 
solid rest4 = boxout and not cylin1out and not cone1out and not cylin2out and not shellend;
 


tlo rest1 -transparent -col=[0,0,1];#air
tlo rest2 -transparent -col=[0,0,1];#air
tlo rest3 -transparent -col=[0,0,1];#air
tlo rest4 -transparent -col=[0,0,1];#air

tlo shell1  -col=[1,0,0];#shell -mur=1 -sig=1.5E+07
tlo shell2  -col=[1,0,0];#shell
tlo shell3  -col=[1,0,0];#shell
tlo shellend  -col=[1,0,0];#shell

