import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
try:
    sys.path.insert(0,"Settings")
except:
    pass
from PlotterSettings import PlotterSettings


#Function definition to edit the tick values to a nice format
def TickFormatter(value,tick_number):
    """
    B.A. Wilson, P.D. Ledger.
    Function to format tick labels for neat plotting

    Args:
        value (float): value at tick
        tick_number (int): No longer used. Originally for even distribution of ticks.

    Returns:
        _type_: _description_
    """
    
    if value==0:
        newvalue = "0"
    elif value==1:
        newvalue = "1"
    elif value==-1:
        newvalue = "-1"
    else:
        for i in range(100):
            if abs(value)<=1:
                if round(abs(value/10**(-i)),2)>=1:
                    power=-i
                    break
            else:
                if round(abs(value/10**(i)),2)<1:
                    power=i-1
                    break
        newvalue=value/(10**power)
        newvalue=str(round(newvalue,2))
        if newvalue[-1]=="0":
            newvalue=newvalue[:-1]
        if newvalue[-1]=="0":
            newvalue=newvalue[:-1]
        if newvalue[-1]==".":
            newvalue=newvalue[:-1]
        newvalue += "e"+str(power)

    return newvalue


