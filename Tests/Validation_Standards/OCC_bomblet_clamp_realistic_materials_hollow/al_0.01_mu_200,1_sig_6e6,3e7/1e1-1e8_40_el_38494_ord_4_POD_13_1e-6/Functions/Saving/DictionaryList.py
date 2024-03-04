#Importing
import os
import sys
from math import floor, log10
import numpy as np
from shutil import copyfile
from zipfile import *

import netgen.meshing as ngmeshing
from ngsolve import Mesh
from .FtoS import *


def DictionaryList(Dictionary, Float):
    ParameterList = []
    for key in Dictionary:
        if key != "air":
            if Float == True:
                newval = FtoS(Dictionary[key])
            else:
                newval = str(Dictionary[key])
                if newval[-1] == "0":
                    newval = newval[:-1]
                if newval[-1] == ".":
                    newval = newval[:-1]
            ParameterList.append(newval)
    ParameterList = ','.join(ParameterList)

    return ParameterList

