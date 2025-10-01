import numpy as np
import shutil

def Add_material(geometry,subdomain_dict):

    # Source and destination
    src = geometry
    dst = geometry[:-4] + "_append.geo"

    # Copy File
    #shutil.copy(src, dst)

    f = open(src, "r")
    f1 = f.readlines()
    count=0
    f2= f1.copy()
    for line in f1:
        count+=1
        # Search for lines where a top level object has been defined
        if line[:3] == "tlo":
            # find the materials and save them in the list
            # Find where the material name starts
            place = line.find("#")
            # Find where the end of the material name is
            if line[-1:] == "\n":
                matend = line.find(" ", place)
                mat = line[place + 1:matend]
            else:
                if line.find(" ", place) != -1:
                    matend = line.find(" ", place)
                    mat = line[place + 1:matend]
                else:
                    mat = line[place + 1:]
            if mat == "air":
                newline=line[0:place-1]+"-material="+mat+" "+line[place-1:]
            else:
                newline=line[0:place-1]+"-material="+subdomain_dict[mat]+" "+line[place-1:]
            f1[count-1]=newline
    f.close()
    f = open(dst, "w")
    f.writelines(f1)
    f.close()

    return
