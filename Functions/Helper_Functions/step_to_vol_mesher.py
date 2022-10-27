from netgen.occ import *


def step_mesher(filename, output_filename='outputstep.vol', add_boundary_layer=False,
                boundary_layer_mat_name='default'):
    """
    James Elgy - 2022
    Function to convert between step format and netgen .vol format.
    Function adds a free space bounding box to the original object and meshes the whole domain before exporting the
    resultant mesh as a .vol file.
    There is the additional option to add boundary layer elements to the mesh boundaries using the add_boundary_layer
    flag.

    :param filename: Step file input filename
    :param output_filename: .vol and .geo files output filename.
    :param add_boundary_layer: bool to control adding additional boundary layer elements. Elements are added inside the
                               object, thus preserving the dimensions.
    :param boundary_layer_mat_name: Material name for the boundary layer elements. If default name matches the main
                                    object.
    :return: geo. OCC geometry object.
    """


    # Loading in and healing step geometry
    geo = OCCGeometry(filename)
    geo = geo.shape.Move((-geo.shape.solids[0].center.x, -geo.shape.solids[0].center.y, -geo.shape.solids[0].center.z))
    # geo.Heal()

    # Assigning material names and boundary conditions
    geo.bc('default')
    geo.mat('mat1')
    geo.solids.name = 'mat1'
    geo.faces.name = 'default'
    geo.maxh = 1

    add_cut = False
    if add_cut is True:
        pos_half = geo.shape - Box(Pnt(0, 100, 100), Pnt(-100, -100, -100))
        neg_half = geo.shape - Box(Pnt(0, 100, 100), Pnt(100, -100, -100))
        geo_rejoined = pos_half + neg_half

        geo_rejoined.bc('default')
        geo_rejoined.mat('mat1')
        geo_rejoined.solids.name = 'mat1'
        geo_rejoined.faces.name = 'default'
        geo_rejoined.maxh = 2 * 3.45

    # Creating boundary box
    bounding_box = Box(Pnt(-1000, -1000, -1000), Pnt(1000, 1000, 1000))
    bounding_box.mat('air')
    bounding_box.bc('outer')
    # bounding_box.maxh = 200
    # bounding_box.faces.name = 'rest'
    # bounding_box.solids.name = 'rest'

    # Adding bounding box to original geometry
    geo2 = OCCGeometry(Glue([geo, bounding_box]))

    if add_cut is True:
        geo2 = OCCGeometry(Glue([geo_rejoined, bounding_box]))

    # Meshing object
    mesh = geo2.GenerateMesh(minh=1)

    if add_boundary_layer is True:
        if boundary_layer_mat_name == 'default':
            mat = 'mat1'
        else:
            mat = boundary_layer_mat_name

        mesh.BoundaryLayer(boundary=".*", thickness=[5e-5], material=mat,
                           domains='mat1', outside=False)

    mesh.Save(output_filename)

    """ GEO FILE GENERATION"""

    # Writing .geo file:
    geo_filename = output_filename[:-4] + '.geo'
    mur = 1
    sigma = 15e6
    if boundary_layer_mat_name != 'default':
        mur_layer = 100
        sigma_layer = 1 * 1e6

    # Creating geo file with same name as .vol file.
    with open(geo_filename, 'w') as file:
        file.write('algebraic3d\n')
        file.write('\n')
        file.write('tlo rest -transparent -col=[0,0,1];#air')
        file.write('\n')
        file.write('tlo mat1 -col=[1,0,0];#mat1 -mur=' + str(mur) + ' -sig=' + str(sigma))

        if boundary_layer_mat_name != 'default':
            file.write('\n')
            file.write(
                'tlo ' + boundary_layer_mat_name + '_layer -col=[1,0,0];#' + boundary_layer_mat_name + ' -mur=' + str(mur_layer) + ' -sig=' + str(sigma_layer))

    return geo


if __name__ == '__main__':
    filename = r'/home/james/Desktop/EinScan_17thAug/Key6_low_detail_remeshed.step'
    output_filename = '../VolFiles/remeshed_matt_key_6_brass_fine.vol'
    geo = step_mesher(filename, output_filename=output_filename, add_boundary_layer=False)
    print('Done')