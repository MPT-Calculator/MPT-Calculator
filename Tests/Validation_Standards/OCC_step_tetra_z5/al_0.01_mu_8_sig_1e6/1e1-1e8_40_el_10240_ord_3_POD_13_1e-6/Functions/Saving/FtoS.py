import os
import sys
from math import floor, log10
import numpy as np
from shutil import copyfile
from zipfile import *

import netgen.meshing as ngmeshing
from ngsolve import Mesh


# Function to edit floats to a nice format
def FtoS(value):
    if value == 0:
        newvalue = "0"
    elif value == 1:
        newvalue = "1"
    elif value == -1:
        newvalue = "-1"
    else:
        for i in range(100):
            if abs(value) <= 1:
                if round(abs(value / 10 ** (-i)), 2) >= 1:
                    power = -i
                    break
            else:
                if round(abs(value / 10 ** (i)), 2) < 1:
                    power = i - 1
                    break
        newvalue = value / (10 ** power)
        newvalue = str(round(newvalue, 2))
        if newvalue[-1] == "0":
            newvalue = newvalue[:-1]
        if newvalue[-1] == "0":
            newvalue = newvalue[:-1]
        if newvalue[-1] == ".":
            newvalue = newvalue[:-1]
        newvalue += "e" + str(power)

    return newvalue
