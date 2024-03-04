########################################################################
#This code was written by Ben Wilson with the supervison of Paul Ledger
#at the ZCCE, Swansea University
#Powered by NETGEN/NGSolve
import inspect
#Additional edits made by James Elgy and Paul Ledger, Keele University, 2022.

########################################################################

# Importing
import sys
from time import time
import numpy as np
import subprocess
import os
from warnings import warn
from shutil import copytree, rmtree

from matplotlib import pyplot as plt

sys.path.insert(0, "Functions")
sys.path.insert(0, "Settings")
from Functions.Helper_Functions.exact_sphere import exact_sphere
from Functions.MeshMaking.Meshmaker import *
from Functions.MeshMaking.VolMatUpdater import *
from Functions.MeshMaking.Generate_From_Python import *
from Settings import *
from Functions.FullSweep.SingleFrequency import *
from Functions.FullSweep.FullSweep import *
from Functions.FullSweep.FullSweepMulti import *
from Functions.POD.PODSweep import *
from Functions.POD.PODSweepMulti import *
from Functions.POD.PODSweepIterative import *
from Functions.Saving.FullSave import *
from Functions.Saving.SingleSave import *
from Functions.Saving.PODSave import *
from Functions.Saving.FolderMaker import *
from Checkvalid import *
from Functions.Helper_Functions.count_prismatic_elements import count_prismatic_elements



def main(h='coarse', order=2, curve_degree=5, start_stop=(), alpha='', geometry='default', frequency_array='default', use_OCC=False,
         use_POD=False, use_parallel=True, use_iterative_POD=False, cpus='default', N_POD_points='default'):
    """
    Main function to run 1D MPT calculator. Some common options have been added as function arguments to make iteration
    easier.

    :param h: int for the mesh size. E.g. setting h=2 will result in MeshSize=2
    :param order: int for the order of the elements.
    :param curve_degree: int for order of curved surface approximation. 5 is usually sufficient.
    :param start_stop: tuple for starting frequency and stopping frequency. e.g. start_stop=(Start, Finish, Points)
    :param alpha: float for Alpha.
    :param geometry: string proxy for GeoFile or OCC py file.
    :param frequency_array: list for explicit override used to specify exact frequencies of interest.
    :param use_OCC: bool for control over using OCC geometry generated via python. When used, the mesh will be
    generated via the OCC package and stored in the VolFiles folder. An associated .geo file is also created containing
    the material names and parameters. In this case, MeshSize in main.py does nothing.
    :param use_POD : bool for control over using POD
    :param use_parallel: bool for using parallel implementation.
    :param use_iterative_POD: bool for control over using adaptive POD mode where POD snapshots are chosen based on
    max error.
    :param cpus: int to overwrite CPUs option in the settings file. Useful when iterating over different  FEM
    discretisations.
    :param N_POD_points: int overwrite to control how many POD snapshots are used when using POD or adaptive POD.

    :return

    Example:
    for p in [1,2,3]:
        print(f'solving for order {p}')
        Return_Dict = main(order=p, geometry='sphere.geo')

    """

    if len(start_stop) == 3:
        OVERWRITE_START_STOP = True
    else:
        OVERWRITE_START_STOP = False

    if type(alpha) == int or type(alpha) == float:
        OVERWRITE_ALPHA = True
    else:
        OVERWRITE_ALPHA = False

    #User Inputs

    #Scaling to be used in the sweep in meters
    if OVERWRITE_ALPHA == False:
        alpha = 1e-3
    #(float) scaling to be applied to the .geo file i.e. if you have defined
    #a sphere of unit radius in a .geo file   alpha = 0.01   would simulate a
    #sphere with a radius of 0.01m ( or 1cm)


    #Geometry
    if use_OCC is True:
        if geometry != 'default':
            OCC_file = geometry
        else:
            # (string) Name of the .py file to be used generate a mesh i.e.
            # "OCC_sphere.py"
            OCC_file = 'OCC_sphere.py'

        alpha = Generate_From_Python(OCC_file, alpha)
        Geometry = OCC_file[:-3] + '.geo'
    else:
        #(string) Name of the .geo file to be used in the frequency sweep i.e.
        # "sphere.geo"
        if geometry != 'default':
            Geometry = geometry
        else:
            Geometry = 'sphere.geo'
    print(Geometry)


    #About the mesh
    #How fine should the mesh be
    MeshSize = 2
    if h != 2:
        MeshSize = h
    #(int 1-5) this defines how fine the mesh should be for regions that do
    #not have maxh values defined for them in the .geo file (1=verycoarse,
    #5=veryfine)

    #The order of the elements in the mesh
    Order = 2
    if order != 2:
        Order = order
    #(int) this defines the order of each of the elements in the mesh

    #About the Frequency sweep (frequencies are in radians per second)
    #Minimum frequency (Powers of 10 i.e Start = 2 => 10**2)
    Start = 1
    #Maximum frequency (Powers of 10 i.e Start = 8 => 10**8)
    Finish = 8

    if OVERWRITE_START_STOP:
        Start = start_stop[0]
        Finish = start_stop[1]
    #(float)

    # (int) Number of logarithmically spaced points in the frequency sweep
    Points = 40

    if OVERWRITE_START_STOP:
        Points = start_stop[2]

    #I only require a single frequency
    Single = False
    #(boolean) True if single frequency is required
    Omega = 10
    #(float) the frequency to be solved if Single = True

    #POD
    #I want to use POD in the frequency sweep
    Pod = False
    if use_POD is True:
        Pod = True
    #(boolean) True if POD is to be used, the number of snapshots can be
    #edited in in the Settings.py file

    #MultiProcessing
    MultiProcessing = True
    if use_parallel is False:
        MultiProcessing = False
    #(boolean) #I have multiple cores at my disposal and have enough spare RAM
    # to run the frequency sweep in parallel (Edit the number of cores to be
    #used in the Settings.py file)

    ########################################################################
    #Main script

    #Load the default settings
    CPUs,BigProblem,PODPoints,PODTol,OldMesh,OldPOD, NumSolverThreads, drop_tol = DefaultSettings()

    if N_POD_points != 'default':
        PODPoints = N_POD_points

    if cpus != 'default':
        CPUs = cpus
    # Here, we overwrite the OldMesh option, since using the OCC geometry will generate a mesh already.
    if use_OCC is True:
        OldMesh = True

    if OldMesh == False:
        #Create the mesh
        print('generating mesh')
        Meshmaker(Geometry,MeshSize)
    else:
        #Check whether to add the material information to the .vol file
        try:
            Materials,mur,sig,inorout,cond,ntags,tags = VolMatUpdater(Geometry,OldMesh)
            ngmesh = ngmeshing.Mesh(dim=3)
            ngmesh.Load("VolFiles/"+Geometry[:-4]+".vol")
            mesh = Mesh("VolFiles/"+Geometry[:-4]+".vol")
            mu_coef = [ mur[mat] for mat in mesh.GetMaterials() ]
        except:
            #Force update to the .vol file
            OldMesh = False



    # For the combination of curved elements and high order elements, one can use the same quadrature rule as one would
    # use for high order elements on flat sided elements ie if the integrand is of degree 2*(order+1) then we just need
    # to use an rule that can integrate up to 2*(order+1) exactly. This is because other approximations are made with
    # fem and the error associated with the numerical integration gets absorbed with the other approximation. The
    # result is care of Ciarlet, Finite element method for elliptic equations chapter 4 pg 178 and Section 4.4. However,
    # this only holds true when considering integals needed for the FEM approximation ie in the bilinear or linear
    # forms.
    #
    # When computing the MPTs as post processing steps, we have integrals not associated with a FEM approximation
    # eg |B| = int_B \dxi or int_B e_i \times \xi \cdot e_j \times \xi d\xi that strongly depend on the geometry order
    # (and not FEM order) and hence we set an integration_order for the post processing that is very conservative and
    # takes both 3*(curve-1) and 2*(order+1) in to account as the degree of the integrand.

    # From testing with larger objects, we observed errors between the integral method and equivalent matrix
    # multiplication method. We suspect NGSolve's integration rule goes wrong for higher degrees (its probably obtained
    # through recursive relationships to get the Gauss points or solving a system and there is too much round-off error
    # for higher degrees). In practice this should be enough for curve=5 and/or order up to 5.
    Integration_Order = np.min( [np.max([2*(Order+1), 3*(curve_degree-1)]), 12] )

    if curve_degree > 5:
        warn('Using a curve degree > 5 may result in inaccurate integration.')

    # Here, we figure out if the mesh contains prismatic elements or not. This is then used when assigning bonus
    # integration orders (0 if prisms, 2 else) when constructing the bilinear and linear forms used in the fast
    # computation of the tensor coefficients using POD.
    # See: https://ngsolve.org/forum/ngspy-forum/1692-question-about-using-curved-surface-elements-with-a-complex-hcurl-discretisation#4572

    # EDIT JAMES 31 May 2023:
    # After some discussion, we've decided that slightly overintegrating the prismatic elements is a small price to pay
    # for consistency of integration order. I.e. we don't really know what order the prisms are integrated with by
    # default in the linear and bilinear forms.
    Order_L2 = 0 # using piecewise constants for material properties.
    N_prisms, N_tets = count_prismatic_elements('./VolFiles/' +Geometry[:-4]+".vol")
    if N_prisms > 0:
        prism_flag = True
        Additional_Int_Order = 2 + Order_L2  # Note that this could be reduced to 1*Oder_L2
    else:
        prism_flag = False
        Additional_Int_Order = 2 + Order_L2

    print(f'Mesh Contains Prisms? {prism_flag}')
    print(f'N Prisms: {N_prisms}, N Tets: {N_tets}')



    #Update the .vol file and create the material dictionaries
    Materials,mur,sig,inorout,cond,ntags,tags = VolMatUpdater(Geometry,OldMesh)

    #create the array of points to be used in the sweep
    Array = np.logspace(Start,Finish,Points)
    if frequency_array != 'default':
        Array = frequency_array
        if len(Array) == 1:
            Single = True
            Omega = float(Array[0])
    PlotPod, PODErrorBars, EddyCurrentTest, vtk_output, Refine, Save_U = AdditionalOutputs()
    SavePOD = False
    if PODErrorBars!=True:
        ErrorTensors=False
    else:
        ErrorTensors=True
    PODArray = np.logspace(Start,Finish,PODPoints)

    # Array = PODArray

    #Create the folders which will be used to save everything
    sweepname = FolderMaker(Geometry, Single, Array, Omega, Pod, PlotPod, PODArray, PODTol, alpha, Order, MeshSize, mur, sig, ErrorTensors, vtk_output, use_OCC)

    # Saving script that calls main() and generating dictionary for returning
    ReturnDict = {}
    ReturnDict['HostScriptFileName'] = inspect.stack()[1].filename

    try:
        copyfile(ReturnDict['HostScriptFileName'], 'Results/' + sweepname + '/Input_files/HostScript.py')
    except FileNotFoundError:
        warn('It looks like the main function was invoked from a jupyter notebook. \nCurrently saving a .ipynb file is done by copying the most recent file in the .ipynb_checkpoints folder. \nUnless you saved the file before running the code, this may not be the correct file.', stacklevel=1)
        path = get_ipython().starting_dir + '/.ipynb_checkpoints/'
        max_mtime = 0
        for dirname, subdirs, files in os.walk(path):
            for fname in files:
                full_path = os.path.join(dirname, fname)
                mtime = os.stat(full_path).st_mtime
                if mtime > max_mtime:
                    max_mtime = mtime
                    max_dir = dirname
                    max_file = fname

        copyfile(path + max_file, 'Results/' + sweepname + '/Input_files/HostScript.ipynb')

    #Run the sweep

    #Check the validity of the eddy-current model for the object
    if EddyCurrentTest == True:
        EddyCurrentTest = Checkvalid(Geometry,Order,alpha,inorout,mur,sig,cond,ntags,tags, curve_degree, Integration_Order, Additional_Int_Order)

    if Single==True:
        if MultiProcessing!=True:
            CPUs = 1
        MPT, EigenValues, N0, elements, ndofs = SingleFrequency(Geometry,Order,alpha,inorout,mur,sig,Omega,CPUs,vtk_output,Refine, Integration_Order,
                                                                Additional_Int_Order, Order_L2, sweepname, drop_tol, curve=curve_degree, num_solver_threads=NumSolverThreads)
        TensorArray = MPT.ravel()
    else:
        if Pod==True:
            if MultiProcessing==True:
                if PlotPod==True:
                    if PODErrorBars==True and use_iterative_POD is False:
                        TensorArray, EigenValues, N0, PODTensors, PODEigenValues, elements, ErrorTensors, ndofs = PODSweepMulti(Geometry,Order,alpha,inorout
                                                                                                                                ,mur,sig,Array,PODArray,PODTol,PlotPod,CPUs,sweepname,SavePOD,
                                                                                                                                PODErrorBars,BigProblem, Integration_Order, Additional_Int_Order,
                                                                                                                                Order_L2, drop_tol, curve=curve_degree, recoverymode=OldPOD, save_U=Save_U,
                                                                                                                                NumSolverThreads=NumSolverThreads)
                    elif use_iterative_POD is True:

                        sweepname_temp = sweepname
                        # Array_Orig = Array
                        TensorArray, EigenValues, N0, PODTensors, PODEigenValues, elements, ErrorTensors, ndofs, PODArray, PODArray_orig, TensorArray_orig, EigenValues_orig, ErrorTensors_orig, PODEigenValues_orig, PODTensors_orig, N_Snaps, Error_Array, Array, Array_Orig = PODSweepIterative(
                            Geometry, Order, alpha, inorout, mur, sig, Array, PODArray,PlotPod, sweepname,
                            SavePOD, PODErrorBars, BigProblem, Integration_Order, Additional_Int_Order, drop_tol, curve=curve_degree, use_parallel=True, cpus=CPUs, save_U=Save_U)

                        sweepname = FolderMaker(Geometry, Single, Array, Omega, Pod, PlotPod, PODArray, PODTol,
                                                alpha, Order, MeshSize, mur, sig, True, vtk_output, use_OCC,
                                                using_interative_POD=True)
                        # Copying files from temporary storage in sweepname_temp to ideal storage in sweepname
                        for item in ['Errors', 'Tensors', 'PODTensors', 'PODEigenValues', 'PODArray', 'FrequencyArray']:
                            for iterations in range(1,len(Error_Array)):
                                os.replace('Results/' + sweepname_temp + f'/Data/{item}_iter{iterations}.npy', 'Results/' + sweepname + f'/Data/{item}_iter{iterations}.npy')

                        for item in ['Imag_Tensor_Coeffs', 'Real_Tensor_Coeffs', 'SVD_Decay']:
                            for iterations in range(1,len(Error_Array)):
                                os.replace('Results/' + sweepname_temp + f'/Graphs/{item}_iter{iterations}.pdf', 'Results/' + sweepname + f'/Graphs/{item}_iter{iterations}.pdf')

                    else:
                        TensorArray, EigenValues, N0, PODTensors, PODEigenValues, elements, ndofs = PODSweepMulti(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,CPUs,
                                                                                                                  sweepname,SavePOD,PODErrorBars,BigProblem, Integration_Order, Additional_Int_Order,
                                                                                                                  Order_L2, drop_tol, curve=curve_degree, recoverymode=OldPOD, NumSolverThreads=NumSolverThreads,
                                                                                                                  save_U=Save_U)
                else:
                    if PODErrorBars==True and use_iterative_POD is False:
                        TensorArray, EigenValues, N0, elements, ErrorTensors, ndofs = PODSweepMulti(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,CPUs,sweepname,
                                                                                                    SavePOD,PODErrorBars,BigProblem, Integration_Order, Additional_Int_Order,Order_L2, drop_tol,
                                                                                                    curve=curve_degree, NumSolverThreads=NumSolverThreads, recoverymode=OldPOD, save_U=Save_U)
                    elif use_iterative_POD is True:
                        sweepname_temp = sweepname
                        # Array_Orig = Array
                        TensorArray, EigenValues, N0, elements, ErrorTensors, ndofs, PODArray, PODArray_orig, TensorArray_orig, EigenValues_orig, ErrorTensors_orig, PODEigenValues_orig, PODTensors_orig, N_Snaps, Error_Array, Array, Array_Orig = PODSweepIterative(
                            Geometry, Order, alpha, inorout, mur, sig, Array, PODArray, PlotPod, sweepname,
                            SavePOD, PODErrorBars, BigProblem, Integration_Order, Additional_Int_Order, drop_tol, curve=curve_degree, use_parallel=True, cpus=CPUs, save_U=Save_U)

                        sweepname = FolderMaker(Geometry, Single, Array, Omega, Pod, PlotPod, PODArray, PODTol,
                                                alpha, Order, MeshSize, mur, sig, True, vtk_output, use_OCC,
                                                using_iterative_POD=True)

                        for item in ['Errors', 'Tensors', 'PODTensors', 'PODEigenValues', 'PODArray', 'FrequencyArray']:
                            for iterations in range(1,len(Error_Array)):
                                os.replace('Results/' + sweepname_temp + f'/Data/{item}_iter{iterations}.npy',
                                           'Results/' + sweepname_temp + f'/Data/{item}_iter{iterations}.npy')

                        for item in ['Imag_Tensor_Coeffs', 'Real_Tensor_Coeffs', 'SVD_Decay']:
                            for iterations in range(1,len(Error_Array)):
                                os.replace('Results/' + sweepname_temp + f'/Graphs/{item}_iter{iterations}.pdf',
                                           'Results/' + sweepname_temp + f'/Graphs/{item}_iter{iterations}.pdf')

                    else:
                        TensorArray, EigenValues, N0, elements, ndofs = PODSweepMulti(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,CPUs,sweepname,SavePOD,
                                                                                      PODErrorBars,BigProblem,Integration_Order, Additional_Int_Order,Order_L2, drop_tol, curve=curve_degree,
                                                                                      prism_flag=prism_flag, NumSolverThreads=NumSolverThreads, recoverymode=OldPOD, save_U=Save_U)
            else:
                if OldPOD is False:
                    if PlotPod==True:

                        if use_iterative_POD is True:
                            sweepname_temp = sweepname
                            TensorArray, EigenValues, N0, PODTensors, PODEigenValues, elements, ErrorTensors, ndofs, PODArray, PODArray_orig, TensorArray_orig, EigenValues_orig, ErrorTensors_orig, PODEigenValues_orig, PODTensors_orig, N_Snaps, Error_Array, Array, Array_Orig = PODSweepIterative(
                                Geometry, Order, alpha, inorout, mur, sig, Array, PODArray, PlotPod, sweepname,
                                SavePOD, PODErrorBars, BigProblem, Integration_Order, Additional_Int_Order, drop_tol, curve=curve_degree, use_parallel=True, cpus=CPUs, save_U=Save_U)

                            sweepname = FolderMaker(Geometry, Single, Array, Omega, Pod, PlotPod, PODArray, PODTol,
                                                    alpha, Order, MeshSize, mur, sig, True, vtk_output, use_OCC,
                                                    using_interative_POD=True)

                            for item in ['Errors', 'Tensors', 'PODTensors', 'PODEigenValues', 'PODArray', 'FrequencyArray']:
                                for iterations in range(1,len(Error_Array)):
                                    os.replace('Results/' + sweepname_temp + f'/Data/{item}_iter{iterations}.npy', 'Results/' + sweepname + f'/Data/{item}_iter{iterations}.npy')

                            for item in ['Imag_Tensor_Coeffs', 'Real_Tensor_Coeffs', 'SVD_Decay']:
                                for iterations in range(1,len(Error_Array)):
                                    os.replace('Results/' + sweepname_temp + f'/Graphs/{item}_iter{iterations}.pdf', 'Results/' + sweepname + f'/Graphs/{item}_iter{iterations}.pdf')

                        else:
                            if PODErrorBars==True:
                                TensorArray, EigenValues, N0, PODTensors, PODEigenValues, elements, ErrorTensors, ndofs = PODSweep(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,
                                                                                                                                   PODTol,PlotPod,sweepname,SavePOD,PODErrorBars,BigProblem,
                                                                                                                                   NumSolverThreads,Integration_Order, Additional_Int_Order, drop_tol,
                                                                                                                                   Order_L2, curve=curve_degree, save_U=Save_U)
                            else:
                                TensorArray, EigenValues, N0, PODTensors, PODEigenValues, elements, ndofs = PODSweep(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,
                                                                                                                     sweepname,SavePOD,PODErrorBars,BigProblem, NumSolverThreads,
                                                                                                                     Integration_Order, Additional_Int_Order, drop_tol, Order_L2, curve=curve_degree,
                                                                                                                     save_U=Save_U)
                    else:
                        if use_iterative_POD is True:
                            sweepname_temp = sweepname
                            Array_Orig = Array
                            TensorArray, EigenValues, N0, elements, ErrorTensors, ndofs, PODArray, PODArray_orig, TensorArray_orig, EigenValues_orig, ErrorTensors_orig, PODEigenValues_orig, PODTensors_orig, N_Snaps, Error_Array, Array, Array_Orig = PODSweepIterative(
                                Geometry, Order, alpha, inorout, mur, sig, Array, PODArray, PlotPod, sweepname,
                                SavePOD, PODErrorBars, BigProblem, Integration_Order, Additional_Int_Order, drop_tol,curve=curve_degree, cpus=CPUs, save_U=Save_U)

                            sweepname = FolderMaker(Geometry, Single, Array, Omega, Pod, PlotPod, PODArray, PODTol,
                                                    alpha, Order, MeshSize, mur, sig, True, vtk_output, use_OCC,
                                                    using_iterative_POD=True)

                            for item in ['Errors', 'Tensors', 'PODTensors', 'PODEigenValues', 'PODArray', 'FrequencyArray']:
                                for iterations in range(1,len(Error_Array)):
                                    os.replace('Results/' + sweepname_temp + f'/Data/{item}_iter{iterations}.npy', 'Results/' + sweepname + f'/Data/{item}_iter{iterations}.npy')

                            for item in ['Imag_Tensor_Coeffs', 'Real_Tensor_Coeffs', 'SVD_Decay']:
                                for iterations in range(1,len(Error_Array)):
                                    os.replace('Results/' + sweepname_temp + f'/Graphs/{item}_iter{iterations}.pdf', 'Results/' + sweepname + f'/Graphs/{item}_iter{iterations}.pdf')

                        else:

                            if PODErrorBars==True:
                                TensorArray, EigenValues, N0, elements, ErrorTensors, ndofs = PODSweep(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,sweepname,SavePOD,
                                                                                                       PODErrorBars,BigProblem, Integration_Order, Additional_Int_Order, Order_L2, drop_tol,
                                                                                                       curve=curve_degree, save_U=Save_U)
                            else:
                                TensorArray, EigenValues, N0, elements, ndofs = PODSweep(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,sweepname,SavePOD,PODErrorBars,
                                                                                         BigProblem, NumSolverThreads, Integration_Order, Additional_Int_Order, Order_L2, drop_tol,
                                                                                         curve=curve_degree, save_U=Save_U)

                else:
                    if PlotPod == True:
                        if PODErrorBars == True:
                            TensorArray, EigenValues, N0, PODTensors, PODEigenValues, elements, ErrorTensors, ndofs = PODSweep(
                                Geometry, Order, alpha, inorout, mur, sig, Array, PODArray, PODTol, PlotPod, sweepname,
                                SavePOD, PODErrorBars, BigProblem, NumSolverThreads, Integration_Order, Additional_Int_Order, Order_L2, drop_tol, recoverymode=OldPOD, curve=curve_degree,
                                save_U=Save_U)
                        else:
                            TensorArray, EigenValues, N0, PODTensors, PODEigenValues, elements, ndofs = PODSweep(
                                Geometry, Order, alpha, inorout, mur, sig, Array, PODArray, PODTol, PlotPod, sweepname,
                                SavePOD, PODErrorBars, BigProblem, NumSolverThreads, Integration_Order, Additional_Int_Order, Order_L2, drop_tol, recoverymode=OldPOD, curve=curve_degree,
                                save_U=Save_U)
                    else:
                        if PODErrorBars == True:
                            TensorArray, EigenValues, N0, elements, ErrorTensors, ndofs = PODSweep(Geometry, Order,
                                                                                                   alpha, inorout, mur,
                                                                                                   sig, Array, PODArray,
                                                                                                   PODTol, PlotPod,
                                                                                                   sweepname, SavePOD,
                                                                                                   PODErrorBars,
                                                                                                   BigProblem, NumSolverThreads, Integration_Order, Additional_Int_Order,
                                                                                                   Order_L2, drop_tol, recoverymode=OldPOD, curve=curve_degree, save_U=Save_U)
                        else:
                            TensorArray, EigenValues, N0, elements, ndofs = PODSweep(Geometry, Order, alpha, inorout,
                                                                                     mur, sig, Array, PODArray, PODTol,
                                                                                     PlotPod, sweepname, SavePOD,
                                                                                     PODErrorBars, BigProblem, NumSolverThreads, Integration_Order, Additional_Int_Order, Order_L2, drop_tol,
                                                                                     recoverymode=OldPOD, curve=curve_degree, save_U=Save_U)


        else:
            if MultiProcessing==True:
                TensorArray, EigenValues, N0, elements, ndofs = FullSweepMulti(Geometry,Order,alpha,inorout,mur,sig,Array,CPUs,BigProblem, NumSolverThreads, Integration_Order,
                                                                               Additional_Int_Order, Order_L2, sweepname, drop_tol, curve=curve_degree)
            else:
                TensorArray, EigenValues, N0, elements, ndofs = FullSweep(Geometry,Order,alpha,inorout,mur,sig,Array,BigProblem, NumSolverThreads, Integration_Order,
                                                                          Additional_Int_Order, Order_L2, sweepname, drop_tol, curve=curve_degree)


    # Constructing invariants:
    # Here we construct the tensor invariants and store them as a Nx3 complex array. Invariants are ordered as
    # [I1(R+N0) + iI1(I), I2(R+N0) + iI2(I), I3(R+N0) + iI3(I)] for each frequency.
    # I1 = tr(A)
    # I2 = (tr(A)^2 -  tr(A^2))/2
    # I3 = det(A)
    if Single is True:
        invariants = np.zeros(3,dtype=complex)
        invariants[0] = np.sum(EigenValues)
        invariants[1] = (EigenValues[0].real * EigenValues[1].real) + (EigenValues[0].real * EigenValues[2].real) + (EigenValues[1].real * EigenValues[2].real)
        invariants[1] += 1j * ((EigenValues[0].imag * EigenValues[1].imag) + (EigenValues[0].imag * EigenValues[2].imag) + (EigenValues[1].imag * EigenValues[2].imag))
        invariants[2] = np.prod(EigenValues.real) + 1j * (np.prod(EigenValues.imag))
    else:
        invariants = np.zeros((len(Array), 3), dtype=complex)
        for f in range(len(Array)):
            invariants[f, 0] = np.sum(EigenValues[f,:])
            invariants[f, 1] = (EigenValues[f,0].real * EigenValues[f,1].real) + (EigenValues[f,0].real * EigenValues[f,2].real) + (EigenValues[f,1].real * EigenValues[f,2].real)
            invariants[f, 1] += 1j*((EigenValues[f,0].imag * EigenValues[f,1].imag) + (EigenValues[f,0].imag * EigenValues[f,2].imag) + (EigenValues[f,1].imag * EigenValues[f,2].imag))
            invariants[f, 2] = np.prod(EigenValues[f,:].real) + 1j*(np.prod(EigenValues[f,:].imag))

    #Plotting and saving
    if Single==True:
        SingleSave(Geometry, Omega, MPT, EigenValues, N0, elements, alpha, Order, MeshSize, mur, sig, EddyCurrentTest, invariants)
    elif PlotPod==True:
        if Pod==True:
            if use_iterative_POD is True:
                PODSave(Geometry, Array, TensorArray, EigenValues, N0, PODTensors, PODEigenValues, PODArray, PODTol, elements, alpha, Order, MeshSize, mur, sig, ErrorTensors,EddyCurrentTest, invariants, using_iterative_POD=True)
                # PODSave(Geometry, Array_Orig, TensorArray_orig, EigenValues_orig, N0, PODTensors_orig, PODEigenValues_orig, PODArray_orig, PODTol, elements, alpha, Order, MeshSize, mur, sig, ErrorTensors_orig,EddyCurrentTest, invariants, using_iterative_POD=False)
            else:
                PODSave(Geometry, Array, TensorArray, EigenValues, N0, PODTensors, PODEigenValues, PODArray, PODTol, elements, alpha, Order, MeshSize, mur, sig, ErrorTensors,EddyCurrentTest, invariants)

        else:
            FullSave(Geometry, Array, TensorArray, EigenValues, N0, Pod, PODArray, PODTol, elements, alpha, Order, MeshSize, mur, sig, ErrorTensors,EddyCurrentTest, invariants)
    else:
        FullSave(Geometry, Array, TensorArray, EigenValues, N0, Pod, PODArray, PODTol, elements, alpha, Order, MeshSize, mur, sig, ErrorTensors,EddyCurrentTest, invariants)



    # Constructing Return Dictionary
    # ReturnDict = {}
    ReturnDict['TensorArray'] = TensorArray
    ReturnDict['EigenValues'] = EigenValues
    ReturnDict['N0'] = N0
    ReturnDict['NElements'] = elements
    ReturnDict['FrequencyArray'] = Array
    ReturnDict['NDOF'] = ndofs

    if EddyCurrentTest is not False:
        ReturnDict['EddyCurrentTest'] = EddyCurrentTest

    if use_POD is True:
        ReturnDict['PODFrequencyArray'] = PODArray
        ReturnDict['PODTensorArray'] = PODTensors
        ReturnDict['PODEigenValues'] = PODEigenValues
        if PODErrorBars is True:
            ReturnDict['PODErrorBars'] = ErrorTensors

        if use_iterative_POD is True:
            ReturnDict['TensorArray_Orig'] = TensorArray_orig
            ReturnDict['PODArray_Orig'] = PODArray_orig
            ReturnDict['PODTensors_Orig'] = PODTensors_orig
            ReturnDict['PODEigenValues_Orig'] = PODEigenValues_orig
            ReturnDict['EigenValues_Orig'] = EigenValues_orig
            ReturnDict['PODErrorBars_Orig'] = ErrorTensors_orig
            ReturnDict['NSnapshots'] = N_Snaps
            ReturnDict['IterativeMaxError'] = Error_Array

    ReturnDict['Invariants'] = invariants
    ReturnDict['SweepName'] = sweepname

    # Copying across folder structure to desired directory:
    # Copies folder structure from results folder to desired folder, and removes left over folder tree.
    FolderStructure = SaverSettings()
    if FolderStructure != 'Default':
        copytree('Results/' + sweepname, FolderStructure + sweepname, dirs_exist_ok=True)
        rmtree('Results/' + sweepname)


    return ReturnDict



def save_all_figures(path, format='png', suffix='', prefix=''):
    """
    Function to save all open figures to disk.
    Files are named as:
    {suffix}{figure_n}{prefix}.{format}
    :param path: path to the desired saving directory.
    :param format: desired file format. pdf, png, jpg
    :param suffix: additional component of the output filename
    :param prefix: additional component of the output filename
    :return:
    """

    if not os.path.isdir(path):
        os.mkdir(path)
    extension = '.' + format
    if format != 'tex':
        for i in plt.get_fignums():
            plt.figure(i)
            filename = prefix + f'figure_{i}' + suffix
            plt.savefig(os.path.join(path, filename) + extension)
    else:
        raise TypeError('Unrecognised file format')



if __name__ == '__main__':

    output = main()

