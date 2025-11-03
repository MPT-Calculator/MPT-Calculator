# If you prefer not to run MPT-Calculator from a Jupyter Notebook (e.g. if working from within a terminal)
# This script provides a basic framework for setting up a simulation
import os
from main import main
from time import time

if __name__ == '__main__':
   start_time = time();
   geometry = "OCC_lego_smthbrick_prism_32.py" # "OCC_box_prism_32.py"#"OCC_test_sphere_prism_32.py"#"OCC_sphere.py"#"OCC_sphere_prism_32.py"#"OCC_sphere_32.py"#"OCC_dualbar.py"#"OCC_cylinder.py" #"OCC_coin.py"##"CSG_Tetra.py"# "CSG_Knife_Knife_Santoku_carbonsteel_copper_rivets.py"
   CPUs=[6,6,6,6,6]
   for order in [0,1,2,3,4]:
       print("solving order=",order)
       Return_Dict = main(geometry=geometry,use_POD=True,use_parallel=False,use_OCC=True,order=order,cpus=CPUs[order],start_stop=(1,12,100))
#   Return_Dict = main(geometry="CSG_Knife_Knife_Santoku_carbonsteel_copper_rivets.py",use_POD=True,use_parallel=True,use_OCC=True,order=2,cpus=2);
   stop_time = time();
