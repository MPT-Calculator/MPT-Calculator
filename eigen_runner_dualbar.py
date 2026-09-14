# If you prefer not to run MPT-Calculator from a Jupyter Notebook (e.g. if working from within a terminal)
# This script provides a basic framework for setting up a simulation
import os
from main import main
from time import time
import numpy as np
import warnings
#warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

if __name__ == '__main__':
   start_time = time();
   geometry = "OCC_dualbar.py"#"CSG_Tetra.py"# "CSG_Knife_Knife_Santoku_carbonsteel_copper_rivets.py"
   CPUs=[6,6,6,6,6,6]
   for order in [2,3]:
       warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")
       print("solving order=",order)
       # We must not use POD here as we don't compute theta1 any more - which do the eigensolve instead
       # Note at present number of eigenmodes is fixed in the eigensolver
       Return_Dict = main(geometry=geometry,use_POD=False,use_parallel=False,use_OCC=True,start_stop=(-3,12,500), MPT_Eigen=True,N_POD_points=50, 
                        MPT_Eigen_From_POD=False, order=order,cpus=CPUs[order],Amp_scale=2*0.001**3,  Time=np.logspace(-6,-1,300))

   stop_time = time();
