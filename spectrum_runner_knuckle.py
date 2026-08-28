# If you prefer not to run MPT-Calculator from a Jupyter Notebook (e.g. if working from within a terminal)
# This script provides a basic framework for setting up a simulation
import os
from main import main
from time import time
import numpy as np

if __name__ == '__main__':
   start_time = time();
   geometry = "CSG_KnuckleDuster_Knuckle_dusters_brass_plated_stainlesssteel.py"# "CSG_Knife_Knife_Santoku_carbonsteel_copper_rivets.py"
   CPUs=[6,6,6,6,6,6,6]
   for order in [3,4]:
       print("solving order=",order)
       Return_Dict = main(geometry=geometry,use_POD=True,use_parallel=False,use_OCC=True,start_stop=(-1,10,3000), curve_degree=5, MPT_Eigen=False, 
                          N_POD_points=50,MPT_Eigen_From_POD=True, order=order,cpus=CPUs[order],  Time=np.logspace(-9,-2,300))

   stop_time = time();
