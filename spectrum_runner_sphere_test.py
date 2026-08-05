# If you prefer not to run MPT-Calculator from a Jupyter Notebook (e.g. if working from within a terminal)
# This script provides a basic framework for setting up a simulation
import os
from main import main
from time import time
import numpy as np

if __name__ == '__main__':
   start_time = time();
   geometry = "OCC_sphere_prism_4.py"#"CSG_Tetra.py"# "CSG_Knife_Knife_Santoku_carbonsteel_copper_rivets.py"
   CPUs=[6,6,6,6,6,6,6]
   PODArray=np.array([1e-3, 7892.91669004,  25374.73864028,  53653.53300996,  93022.35270049,
       143562.46349437, 205300.95530173, 278248.54603154, 362410.08502918,
       457788.00535374, 564383.63050264, 682197.72835656, 1e12])
   for order in [3]:
       print("solving order=",order)
       #Return_Dict = main(geometry=geometry,use_POD=False,use_parallel=False,use_OCC=True,start_stop=(-3,12,500), MPT_Eigen=False,curve_degree=5,N_POD_points=13,MPT_Eigen_From_POD=True, order=order,cpus=CPUs[order],Amp_scale=2*np.pi*0.01**3,  Time=np.logspace(-6,-1,300))
       for N in [500,1000,3000]:
           Return_Dict = main(geometry=geometry,use_POD=True, use_parallel=False, use_OCC=True, start_stop=(-3,12,N), MPT_Eigen=False,curve_degree=5,N_POD_points=13,MPT_Eigen_From_POD=True, SingleSVD=False, order=order, PODArray='default',cpus=CPUs[order], Amp_scale=2*np.pi*0.01**3,  Time=np.logspace(-6,-1,300))

   stop_time = time();
