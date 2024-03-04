from ..Saving.FtoS import *
from ..Saving.DictionaryList import *
from ngsolve import *


def generate_VTK(mesh, fes2, Output, Omega, sigma, sweepname, alpha, dom_nrs_metal, Refine=True, object_name='default'):
# Create the VTK output if required
    print(' creating vtk output', end='\r')
    ThetaE1 = GridFunction(fes2)
    ThetaE2 = GridFunction(fes2)
    ThetaE3 = GridFunction(fes2)
    ThetaE1.vec.FV().NumPy()[:] = Output[0]
    ThetaE2.vec.FV().NumPy()[:] = Output[1]
    ThetaE3.vec.FV().NumPy()[:] = Output[2]
    E1Mag = CoefficientFunction(
        sqrt(InnerProduct(ThetaE1.real, ThetaE1.real) + InnerProduct(ThetaE1.imag, ThetaE1.imag)))
    E2Mag = CoefficientFunction(
        sqrt(InnerProduct(ThetaE2.real, ThetaE2.real) + InnerProduct(ThetaE2.imag, ThetaE2.imag)))
    E3Mag = CoefficientFunction(
        sqrt(InnerProduct(ThetaE3.real, ThetaE3.real) + InnerProduct(ThetaE3.imag, ThetaE3.imag)))

    Sols = []
    Sols.append(dom_nrs_metal)
    Sols.append((ThetaE1 * 1j * Omega * sigma).real)
    Sols.append((ThetaE1 * 1j * Omega * sigma).imag)
    Sols.append((ThetaE2 * 1j * Omega * sigma).real)
    Sols.append((ThetaE2 * 1j * Omega * sigma).imag)
    Sols.append((ThetaE3 * 1j * Omega * sigma).real)
    Sols.append((ThetaE3 * 1j * Omega * sigma).imag)
    Sols.append(E1Mag * Omega * sigma)
    Sols.append(E2Mag * Omega * sigma)
    Sols.append(E3Mag * Omega * sigma)
    Sols.append(curl(ThetaE1).real)
    Sols.append(curl(ThetaE2).real)
    Sols.append(curl(ThetaE3).real)
    Sols.append(curl(ThetaE1).imag)
    Sols.append(curl(ThetaE2).imag)
    Sols.append(curl(ThetaE3).imag)

    # Creating Save Name:
   # strmur = DictionaryList(mur, False)
   # strsig = DictionaryList(sig, True)
    savename = sweepname + f'{Omega}'
    subs = 0
    if Refine == True:
        subs = 3
    vtk = VTKOutput(ma=mesh, coefs=Sols,
                    names=["Object", "E1real", "E1imag", "E2real", "E2imag", "E3real", "E3imag",
                           "E1Mag","E2Mag", "E3Mag",
                           "CurlThetaE1real", "CurlThetaE2real", "CurlThetaE3real",
                           "CurlThetaE1imag", "CurlThetaE2imag", "CurlThetaE3imag"], filename=savename, subdivision=subs, legacy=True)
    vtk.Do()

    # Compressing vtk output and sending to zip file:
    zipObj = ZipFile(savename + 'VTU.zip', 'w', ZIP_DEFLATED)
    zipObj.write(savename + '.vtk', os.path.basename(savename + '.vtk'))
    zipObj.close()
    os.remove(savename + '.vtk')
    print(' vtk output created     ')
