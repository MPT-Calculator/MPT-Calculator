import subprocess
import runpy
from netgen.csg import *


# Function definition which edits the created .vol file into a format which can used by the for the code
# Input -the name of the .geo file used in the sweep (string)
# Outputs -a list of material names in the geo file
#        -a dictionary of the relative permeabilities for the materials
#        -a dictionary of the conductivities for the materials
#        -a dictionary of which materials are considered to be free space
def VolMatUpdater(Geometry, OldMesh):
    # Remove the .geo part of the file extention
    objname = Geometry[:-4]

    # This part creates a list of materials in the order they appear
    # Create how the materials will be saved
    matlist = []
    orderedmatlist = []
    murlist = []
    siglist = []
    inout = []
    condlist=[]

    # Read the .geo file
    f = open("GeoFiles/" + Geometry, "r")
    f1 = f.readlines()
    for line in f1:
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
            # Add the material name to the list
            orderedmatlist.append(mat)
            # Check whether we've found this material before
            if orderedmatlist.count(mat) == 1 and mat != "air":
                # find the properites for the materials
                # Check how the line ends
                if line[-1:] == "\n":
                    # Check if the line ends "_\n"
                    if line[-2] == " ":
                        if line.find("-mur=") != -1:
                            murplace = line.find("-mur=")
                            murend = line.find(" ", murplace)
                            mur = float(line[murplace + 5:murend])
                            murlist.append(mur)
                        if line.find("-sig=") != -1:
                            sigplace = line.find("-sig=")
                            sigend = line.find(" ", sigplace)
                            sig = float(line[sigplace + 5:sigend])
                            siglist.append(sig)
                    # Line ends in some sort of information
                    else:
                        if line.find("-mur=") != -1:
                            murplace = line.find("-mur=")
                            murend = line.find(" ", murplace)
                            mur = float(line[murplace + 5:murend])
                            murlist.append(mur)
                        if line.find("-sig=") != -1:
                            sigplace = line.find("-sig=")
                            sigend = line.find("\n", sigplace)
                            sig = float(line[sigplace + 5:sigend])
                            siglist.append(sig)
                # must be the last line in the script but ends in a space
                elif line[len(line) - 1] == " ":
                    if line.find("-mur=") != -1:
                        murplace = line.find("-mur=")
                        murend = line.find(" ", murplace)
                        mur = float(line[murplace + 5:murend])
                        murlist.append(mur)
                    if line.find("-sig=") != -1:
                        sigplace = line.find("-sig=")
                        sigend = line.find(" ", sigplace)
                        sig = float(line[sigplace + 5:sigend])
                        siglist.append(sig)
                # must be the last line in the script but ends in some sort of information
                else:
                    if line.find("-mur=") != -1:
                        murplace = line.find("-mur=")
                        murend = line.find(" ", murplace)
                        mur = float(line[murplace + 5:murend])
                        murlist.append(mur)
                    if line.find("-sig=") != -1:
                        sigplace = line.find("-sig=")
                        sig = float(line[sigplace + 5:])
                        siglist.append(sig)
            elif orderedmatlist.count(mat) == 1 and mat == "air":
                murlist.append(1)
                siglist.append(0)

    # Reorder the list so each material just appears once
    for mat in orderedmatlist:
        if mat not in matlist:
            matlist.append(mat)
    # decide in or out
    for mat in matlist:
        if mat == "air":
            inout.append(0)
        else:
            inout.append(1)
#   Create a list of the different materials and tags for each subdomain
#   so that each can be identified seperately
    tags =[]
    ntags = 0
    for mat in matlist:
        flag =0
        for n in range(1,ntags):
            print(n,tags[n-1],mat)
            if mat==tags[n-1]:
                condlist.append(n-1)
                flag=1
        if flag==0:
            # tag not found
            tags.append(mat)
            condlist.append(ntags)
            ntags = ntags +1
    print(condlist,ntags,tags)


    f.close()

    # extract the number of boundaries and the outer boundaries
    # Read the .vol file
    f = open("VolFiles/" + objname + ".vol", "r")
    f1 = f.readlines()
    # Find the line where it says how many surface elements there are
    for line in f1:
        if line[:-1] == "surfaceelements" or line[:-1] == "surfaceelementsuv":
            linenum = f1.index(line)
            break
    surfnumstr = f1[linenum + 1]
    surfnum = int(surfnumstr)
    # Set up where to save the outer edges and a counter for how many edges there are in total
    maxbound = 0
    edgelist = []
    for i in range(surfnum):
        line = f1[linenum + 2 + i]
        # Segment the line to easily take each column.
        segline = line.split(" ")
        # Search for outer edges and add them to the list
        if segline[4] == "0":
            if int(segline[2]) not in edgelist:
                edgelist.append(int(segline[2]))
        # find the boundary with the highest number
        if int(segline[2]) > maxbound:
            maxbound = int(segline[2])
    f.close()

    if OldMesh == False:
        # Create the new lines which are to be added to the .vol file
        # define how many regions there are
        materials = len(orderedmatlist)
        # Create the lines to be written in as a list
        # materials
        newlines = ['materials\n']
        newlines.append(str(materials) + '\n')
        for i in range(materials):
            newlines.append(str(i + 1) + ' ' + orderedmatlist[i] + '\n')
        newlines.append('\n')
        newlines.append('\n')
        # bcnames
        newlines.append('bcnames\n')
        newlines.append(str(maxbound) + '\n')
        for i in range(maxbound):
            if i + 1 < 10:
                if i + 1 in edgelist:
                    newlines.append(str(i + 1) + '   outer\n')
                else:
                    newlines.append(str(i + 1) + '   default\n')
            elif i + 1 < 100:
                if i + 1 in edgelist:
                    newlines.append(str(i + 1) + '  outer\n')
                else:
                    newlines.append(str(i + 1) + '  default\n')
            else:
                if i + 1 in edgelist:
                    newlines.append(str(i + 1) + ' outer\n')
                else:
                    newlines.append(str(i + 1) + ' default\n')
        newlines.append('\n')
        newlines.append('\n')

        # Find where the lines should be added
        f = open("VolFiles/" + objname + ".vol", "r")
        f1 = f.readlines()
        # Find the line where it says how many surface elements there are
        for line in f1:
            if line[:-1] == "points":
                linenum = f1.index(line)
                break
        pointnumstr = f1[linenum + 1]
        pointnum = int(pointnumstr)
        firsthalf = f1[:linenum + pointnum + 2]
        secondhalf = f1[linenum + pointnum + 2:]

        # Stick the lists together
        newfile = firsthalf + newlines + secondhalf
        f.close()
        f = open("VolFiles/" + objname + ".vol", "w")
        for line in newfile:
            f.write(line)
        f.close()

    inorout = dict(zip(matlist, inout))
    mur = dict(zip(matlist, murlist))
    sig = dict(zip(matlist, siglist))
    cond=dict(zip(matlist,condlist))


    return matlist, mur, sig, inorout, cond, ntags, tags
