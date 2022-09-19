import netgen.meshing as ngmeshing
import netgen.csg as csg
import ngsolve as ngsol
import netgen.occ as ngocc

def add_truncated_region_to_mesh(input_filename, output_filename):

    # Takes a surface desciption of an object (best achieved through first reading
    #a STL/Step file in to netgen and generating a mesh) and then adds a box and creates
    #a mesh of the object and region between the box and the object.

    # Specify a geometry or a volume file from which the surface description should be extracted
    # option = 1 # Generate a surface mesh given a step file
    #            -preferred option is 3 as it has greater control and some built in geometry fixes.
    # option = 2 # Generate a surface mesh given a STL file
    #            - not reading from STL for some reason (best to use graphical interface to first get volume mesh from STL)
    # option = 3 # Generate a surface mesh from a volume mesh file

    # Input filename
    Filename= input_filename
    option=3

    # Default output filename output.vol
    # Ready to be used with MPT-Calculator

    # Set box size
    box=1000

    # generate sphere and mesh it
    #geo2 = CSGeometry()
    #geo2.Add (Sphere (Pnt(0,0,0), 1))
    #m3 = geo2.GenerateMesh (maxh=0.1)
    #m3.Save("sphere.vol")

    ###############################################################################
    # Generate an object from a step file
    if option==1:
        geo = ngocc.OCCGeometry(Filename)
        geo.Glue()
        ngmesh2 = ngmeshing.Mesh(geo.GenerateMesh())

    ###############################################################################
    # Generate an object from a STL file

    elif option==2:
        geo = ngocc.OCCGeometry(Filename)
        geo.Glue()
        ngmesh2 = ngmeshing.Mesh(geo.GenerateMesh())

    ###############################################################################
    elif option==3:
        #Load an object
    # This is the full netgen mesh class - that we need to use here!
        ngmesh2 = ngmeshing.Mesh(dim=3)
        ngmesh2.Load(Filename)
    # This is the NGSolve wrapper class around the full Netgen mesh class
        m2 = ngsol.Mesh(Filename)
        print("Loaded mesh")
    #print(dir(ngmesh2))
    else:
        print("Invalid option")
        quit()

    ###############################################################################

    # generate brick and mesh it
    geo1 = csg.CSGeometry()
    geo1.Add (csg.OrthoBrick( csg.Pnt(-box,-box,-box), csg.Pnt(box,box,box) ))
    m1 = geo1.GenerateMesh (maxh=1000)
    # m1.Refine()
    print(dir(m1.Elements2D))

    ###############################################################################


    # create an empty mesh
    ngmesh = ngmeshing.Mesh()

    # a face-descriptor stores properties associated with a set of surface elements
    # bc .. boundary condition marker,
    # domin/domout .. domain-number in front/back of surface elements (0 = void),
    # surfnr .. number of the surface described by the face-descriptor

    fd_outside = ngmesh.Add (ngmeshing.FaceDescriptor(bc=1,domin=1,surfnr=1))
    fd_inside = ngmesh.Add (ngmeshing.FaceDescriptor(bc=2,domin=2,domout=1,surfnr=2))
    orderedmatlist=["air","mat1"]
    # copy all boundary points from first mesh to new mesh.
    # pmap1 maps point-numbers from old to new mesh
    pmap1 = { }
    for e in m1.Elements2D():
        for v in e.vertices:
            if (v not in pmap1):
                pmap1[v] = ngmesh.Add (m1[v])


    # copy surface elements from first mesh to new mesh
    # we have to map point-numbers:

    for e in m1.Elements2D():

        ngmesh.Add (ngmeshing.Element2D (fd_outside, [pmap1[v] for v in e.vertices]))

    # same for the second mesh:
    pmap2 = { }
    for e in ngmesh2.Elements2D():
        for v in e.vertices:
            if (v not in pmap2):
                pmap2[v] = ngmesh.Add (ngmesh2[v])

    for e in ngmesh2.Elements2D():
        ngmesh.Add (ngmeshing.Element2D (fd_inside, [pmap2[v] for v in e.vertices]))

    ngmesh.GenerateVolumeMesh()
    import ngsolve
    mesh = ngsolve.Mesh(ngmesh)
    #Draw(mesh)

    ngmesh.Save(output_filename)

    f=open(output_filename,"r")
    f1 = f.readlines()
    #Find the line where it says how many surface elements there are
    for line in f1:
        #if line[:-1]=="surfaceelements":
        if line[:-1]=="surfaceelements" or line[:-1]=="surfaceelementsuv":
            linenum=f1.index(line)
            break
    surfnumstr=f1[linenum+1]
    surfnum=int(surfnumstr)
    #Set up where to save the outer edges and a counter for how many edges there are in total
    maxbound=0
    edgelist=[]
    for i in range(surfnum):
        line=f1[linenum+2+i]
        #Segment the line to easily take each column.
        segline=line.split(" ")
        #Search for outer edges and add them to the list
        if segline[4]=="0":
            if int(segline[2]) not in edgelist:
                edgelist.append(int(segline[2]))
        #find the boundary with the highest number
        if int(segline[2])>maxbound:
            maxbound=int(segline[2])
    f.close()

    #Create the new lines which are to be added to the .vol file
    #define how many regions there are
    materials=len(orderedmatlist)
    #Create the lines to be written in as a list
    #materials
    newlines=['materials\n']
    newlines.append(str(materials)+'\n')
    for i in range(materials):
        newlines.append(str(i+1)+' '+orderedmatlist[i]+'\n')
    newlines.append('\n')
    newlines.append('\n')
    #bcnames
    newlines.append('bcnames\n')
    newlines.append(str(maxbound)+'\n')
    for i in range(maxbound):
        if i+1<10:
            if i+1 in edgelist:
                newlines.append(str(i+1)+'   outer\n')
            else:
                newlines.append(str(i+1)+'   default\n')
        elif i+1<100:
            if i+1 in edgelist:
                newlines.append(str(i+1)+'  outer\n')
            else:
                newlines.append(str(i+1)+'  default\n')
        else:
            if i+1 in edgelist:
                newlines.append(str(i+1)+' outer\n')
            else:
                newlines.append(str(i+1)+' default\n')
    newlines.append('\n')
    newlines.append('\n')



    #Find where the lines should be added
    f=open(output_filename,"r")
    f1 = f.readlines()
    #Find the line where it says how many surface elements there are
    for line in f1:
        if line[:-1]=="points":
            linenum=f1.index(line)
            break
    pointnumstr=f1[linenum+1]
    pointnum=int(pointnumstr)
    firsthalf=f1[:linenum+pointnum+2]
    secondhalf=f1[linenum+pointnum+2:]

    #Stick the lists together
    newfile=firsthalf+newlines+secondhalf
    f.close()
    f=open(output_filename,"w")
    for line in newfile:
        f.write(line)
    f.close()


if __name__ == '__main__':
    add_truncated_region_to_mesh(r'/home/james/Desktop/220704_EinScan_Day2/Two_P_Coin/Two_P_Coin/2p_high_detail_scan1_orientated.vol', r'../VolFiles/Einscan_2p.vol')