import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.optimize import minimize


# function to obtain PS and Hill tensors
def PStensor(a,b,c,alpha,mur,vol):
    x1=b/a
    x2=c/a
    IA=quad(A,1,np.inf,args=(x1,x2))[0]
    IB=quad(B,1,np.inf,args=(x1,x2))[0]
    IC=quad(C,1,np.inf,args=(x1,x2))[0]
    print(IA)

    IA+IB+IC
    print("PS Tensor")
    PS=alpha**3*(mur-1)*vol*np.array(((1./(1.+IA*(mur-1)),0,0),(0,1./(1.+IB*(mur-1)),0),(0,0,1./(1.+IC*(mur-1)))))
    print(PS)
    Hill=np.linalg.inv(PS)*alpha**3*vol-np.eye(3)/(mur-1)
    print("Hill Tensor")
    print(Hill)
    if np.abs(np.trace(Hill)-1.) > 1e-12:
        print("Trace of Hill tensor is not 1",np.trace(Hill))
    return PS,Hill

def A(t,x1,x2):
    return x1*x2*1./t**2/np.sqrt(t**2-1+x1**2)/np.sqrt(t**2-1+x2**2)

def B(t,x1,x2):
    return x1*x2*1./(t**2-1+x1**2)**(3/2)/np.sqrt(t**2-1+x2**2)

def C(t,x1,x2):
    return x1*x2/np.sqrt(t**2-1+x1**2)/(t**2-1+x2**2)**(3/2)

# FUnction to find equivalent ellipsoid given Hill tensor
def find_ellipsoid(Hill,vol,tol=None):
    # Solve the nonlinear problem IB/IA-Hill(2,2)/Hill(1,1)=0, IC/IA-Hill(3,3)/Hill(1,1)=0,
    # for b/a and c/a
    rhs=np.array((Hill[1,1]/Hill[0,0],Hill[2,2]/Hill[0,0]))
    x0=np.array((1,1))
    #x=fsolve(nonlin,x0,args=(rhs))
    if tol == None:
        res=minimize(nonlin,x0,args=(rhs),bounds=((1-10,1e6),(1e-10,1e6)),method="L-BFGS-B")
    else:
        res=minimize(nonlin,x0,args=(rhs),bounds=((1-10,1e6),(1e-10,1e6)),method="L-BFGS-B",options={"gtol":tol})
    x=res.x
    print("solution to min problem",x)
    if x[0]<0 or x[1]<0:
        # try again
        x0=np.array((0.1,0.1))
        #x=fsolve(nonlin,x0,args=(rhs))
        if tol == None:
            res=minimize(nonlin,x0,args=(rhs),bounds=((1-10,1e6),(1e-10,1e6)),method="L-BFGS-B")
        else:
            res=minimize(nonlin,x0,args=(rhs),bounds=((1-10,1e6),(1e-10,1e6)),method="L-BFGS-B",options={"gtol":tol})
        x=res.x
        print("solution to min problem",x)
        if x[0]<0 or x[1]<0:
            # try again
            x0=np.array((0.01,0.01))
            #x=fsolve(nonlin,x0,args=(rhs))
            if tol == None:
                res=minimize(nonlin,x0,args=(rhs),bounds=((1-10,1e6),(1e-10,1e6)),method="L-BFGS-B")
            else:
                res=minimize(nonlin,x0,args=(rhs),bounds=((1-10,1e6),(1e-10,1e6)),method="L-BFGS-B",options={"gtol":tol})
            x=res.x
            print("solution to min problem",x)
    # x is x[0]=b/a, x[1]=c/a , x[0] * x[1] = b*c / a^2 and so
    # vol = 4/3 * pi a*b*c = 4/3 * pi *a^3 *x[0] *x[1]
    aout=(vol*3./4./np.pi/x[0]/x[1])**(1/3);
    bout=x[0]*aout;
    cout=x[1]*aout;
    elip=-np.sort(-np.array((aout,bout,cout)))
    print("Ellipsoid found is ",elip)
    return elip
def nonlin(x,rhs):
    x1=x[0];
    x2=x[1];
    IA=quad(A,1,np.inf,args=(x1,x2))[0]
    IB=quad(B,1,np.inf,args=(x1,x2))[0]
    IC=quad(C,1,np.inf,args=(x1,x2))[0]
    return np.linalg.norm(np.array((IC/IA-rhs[0], IB/IA-rhs[1])))
