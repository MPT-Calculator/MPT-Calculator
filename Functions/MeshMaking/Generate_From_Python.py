import subprocess
import runpy
from netgen.csg import *


def Generate_From_Python(OCC_file, alpha):
    """
    James Elgy - 2022
    Function to generate from python script using OCC geometry.
    Function also generates associated .geo file in order to comply with the rest of MPT-Calculator.
    
    In MPT-Calculator, we require that OCC geometries are defined in a .py python file (See examples in OCC_Geometry folder).
    The script is then run using runpy in a fresh module namespace.
    All valid python is enabled, and this may be a security concern for some users.
    
    Args:
        OCC_file (str): OCC file name. E.g. OCC_Sphere.py
        alpha (float): object scaling.
    
    Returns:
        alpha (float): Alpha may be updated as part of the OCC script. We therefore return alpha here.
    
    
    
    """

    alpha_orig = alpha

    out = runpy.run_path(f'OCC_Geometry/{OCC_file}')
    mur = out['mur']
    sigma = out['sigma']
    mat_names = out['material_name']

    try:
        out['alpha']
        alpha = out['alpha']
        print(f'Updated alpha from OCC file. Alpha={alpha}')
    except:
        alpha = alpha_orig

    # Writing associated .geo file. This is to maintain compatability with the existing MPT-Calculator.
    with open('GeoFiles/' + OCC_file[:-3] + '.geo', 'w') as file:
        file.write('algebraic3d\n')
        file.write('\n')
        file.write('tlo rest -transparent -col=[0,0,1];#air')
        file.write('\n')

        # if type(object_name) is list:
        #     for obj, obj_mur, obj_sigma, mat in zip(object_name, mur, sigma, mat_name):
        #         file.write(f'tlo {obj} -col=[1,0,0];#{mat} -mur=' + str(obj_mur) + ' -sig=' + str(obj_sigma) + '\n')
        # else:
        #     file.write(f'tlo {object_name} -col=[1,0,0];#{mat_name} -mur=' + str(mur) + ' -sig=' + str(sigma))
        count = 1
        for mat_index in range(1, out['nmesh'].GetNDomains()+1):
            mat_name = out['nmesh'].GetMaterial(mat_index)
            if mat_name != 'air':
                index = mat_names.index(mat_name)
                file.write(f'tlo region{count} -col=[1,0,0];#{mat_name} -mur=' + str(mur[index]) + ' -sig=' + str(sigma[index]) + '\n')
                count += 1

    return alpha