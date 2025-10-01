import os
from main import main
from time import time

if __name__ == '__main__':
   start_time = time();
   geometry = 'OCC_test_key_4_nomag.py'
   test_results = main(geometry=geometry, order=3, use_OCC=True, use_POD=True, alpha=1e-3,cpus=2)
   stop_time = time();
