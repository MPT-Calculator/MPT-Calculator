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
   for order in [5]:
       print("solving order=",order)
       Return_Dict = main(geometry=geometry,use_POD=True,use_parallel=False,use_OCC=True,start_stop=(-3,12,500), MPT_Eigen=False, 
                        MPT_Eigen_From_POD=True, N_POD_points=50, order=order,cpus=CPUs[order],Amp_scale=2*np.pi*0.01**3,  Time=np.logspace(-6,-1,300))

   stop_time = time();
