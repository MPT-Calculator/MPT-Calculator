from math import sqrt

from ngsolve import Mesh as NGSolveMesh

from netgen.meshing import (
    Mesh as NetgenMesh,
    MeshPoint,
    Element2D,
    Element3D,
    FaceDescriptor,
    Pnt,
)


def _element_faces(vertices):
    """
    Faces of tetrahedral and triangular-prismatic elements.
    Used only to determine material adjacency.
    """

    v = list(vertices)

    if len(v) == 4:
        # tetrahedron
        return [
            (v[0], v[1], v[2]),
            (v[0], v[1], v[3]),
            (v[0], v[2], v[3]),
            (v[1], v[2], v[3]),
        ]

    elif len(v) == 6:
        # triangular prism
        return [
            # triangular ends
            (v[0], v[1], v[2]),
            (v[3], v[4], v[5]),

            # quadrilateral sides
            (v[0], v[1], v[4], v[3]),
            (v[1], v[2], v[5], v[4]),
            (v[2], v[0], v[3], v[5]),
        ]

    else:
        raise NotImplementedError(
            f"Unsupported 3D element with {len(v)} vertices"
        )


def DeformedMesh(mesh, V, tau=1.0, verbose=True):
    """
    Construct a new linear 3D Netgen/NGSolve mesh

        x_new = x + tau * V(x)

    while preserving the original vertex numbering.

    Supports:
        * multiple materials
        * tetrahedral elements
        * triangular prisms
        * triangular/quadrilateral interface elements

    Returns
    -------
    mesh_new, info
    """

    if mesh.dim != 3:
        raise ValueError("DeformedMesh requires a 3D mesh")

    old = mesh.ngmesh

    volume_elements = list(old.Elements3D())
    surface_elements = list(old.Elements2D())

    new = NetgenMesh(dim=3)

    # ============================================================
    # 1. Create displaced vertices
    #
    # IMPORTANT:
    #
    # mesh.vertices is iterated in original vertex-number order.
    # Therefore:
    #
    #       old vertex 0 -> new vertex 0
    #       old vertex 1 -> new vertex 1
    #       ...
    #
    # and Netgen:
    #
    #       old PointId 1 -> new PointId 1
    #       old PointId 2 -> new PointId 2
    #       ...
    # ============================================================

    new_point_ids = []

    maximum_displacement = 0.0
    maximum_displacement_data = None

    for v in mesh.vertices:

        p = mesh[v].point

        px = float(p[0])
        py = float(p[1])
        pz = float(p[2])

        # Evaluate displacement on ORIGINAL mesh
        d = V(mesh(px, py, pz))

        vx = float(d[0])
        vy = float(d[1])
        vz = float(d[2])

        dx = tau * vx
        dy = tau * vy
        dz = tau * vz

        xnew = px + dx
        ynew = py + dy
        znew = pz + dz

        new_pid = new.Add(
            MeshPoint(
                Pnt(xnew, ynew, znew)
            )
        )

        new_point_ids.append(new_pid)

        displacement = sqrt(
            dx*dx + dy*dy + dz*dz
        )

        if displacement > maximum_displacement:

            maximum_displacement = displacement

            maximum_displacement_data = {
                "vertex": v.nr,
                "old": (px, py, pz),
                "V": (vx, vy, vz),
                "new": (xnew, ynew, znew),
            }

    # ============================================================
    # 2. Convert an old element's PointIds to the corresponding
    #    new PointIds.
    #
    # Netgen PointId.nr is 1-based:
    #
    # pid.nr == 1 -> new_point_ids[0]
    # ============================================================

    def mapped_vertices(element):

        return [
            new_point_ids[pid.nr - 1]
            for pid in element.vertices
        ]

    # ============================================================
    # 3. Copy material names
    # ============================================================

    materials = list(mesh.GetMaterials())

    for index, name in enumerate(materials, start=1):
        new.SetMaterial(index, name)

    # ============================================================
    # 4. Determine adjacent materials for every volume face
    # ============================================================

    face_to_materials = {}

    element_counts = {
        "tetrahedra": 0,
        "prisms": 0,
    }

    for element in volume_elements:

        vertices = list(element.vertices)
        material = int(element.index)

        if len(vertices) == 4:
            element_counts["tetrahedra"] += 1

        elif len(vertices) == 6:
            element_counts["prisms"] += 1

        for face in _element_faces(vertices):

            key = frozenset(face)

            if key not in face_to_materials:
                face_to_materials[key] = set()

            face_to_materials[key].add(material)

    # ============================================================
    # 5. Determine domin/domout for each surface descriptor
    # ============================================================

    descriptor_domains = {}

    for surface_element in surface_elements:

        fd_index = int(surface_element.index)

        key = frozenset(
            surface_element.vertices
        )

        adjacent = sorted(
            face_to_materials.get(key, set())
        )

        if len(adjacent) == 1:

            # exterior boundary
            domains = (
                adjacent[0],
                0,
            )

        elif len(adjacent) == 2:

            # material interface
            domains = (
                adjacent[0],
                adjacent[1],
            )

        elif len(adjacent) == 0:

            raise RuntimeError(
                "Could not match surface element "
                f"with FaceDescriptor {fd_index} "
                "to any volume-element face."
            )

        else:

            raise RuntimeError(
                "Surface element is adjacent to "
                f"{len(adjacent)} materials: {adjacent}"
            )

        if fd_index in descriptor_domains:

            previous = descriptor_domains[fd_index]

            # We do not know the original orientation,
            # so compare unordered domain pairs.
            if set(previous) != set(domains):

                raise RuntimeError(
                    f"FaceDescriptor {fd_index} has "
                    "inconsistent material adjacency:\n"
                    f"    {previous}\n"
                    f"    {domains}"
                )

        else:

            descriptor_domains[fd_index] = domains

    # ============================================================
    # 6. Create FaceDescriptors BEFORE Element2D objects
    # ============================================================

    if descriptor_domains:

        max_fd = max(descriptor_domains)

        for fd_index in range(1, max_fd + 1):

            if fd_index not in descriptor_domains:
                raise RuntimeError(
                    "FaceDescriptor numbering is not contiguous: "
                    f"descriptor {fd_index} is missing"
                )

            domin, domout = descriptor_domains[
                fd_index
            ]

            new.Add(
                FaceDescriptor(
                    surfnr=fd_index,
                    domin=domin,
                    domout=domout,
                    bc=fd_index,
                )
            )

    # ============================================================
    # 7. Copy boundary names
    # ============================================================

    boundaries = list(mesh.GetBoundaries())

    for index, name in enumerate(boundaries):

        try:
            new.SetBCName(index, name)

        except Exception as error:

            if verbose:
                print(
                    f"Warning: could not set BC name "
                    f"{index}: {name}: {error}"
                )

    # ============================================================
    # 8. Copy volume elements
    #
    # Material number is retained.
    # Vertex numbers are retained.
    # ============================================================

    for element in volume_elements:

        new.Add(
            Element3D(
                int(element.index),
                mapped_vertices(element),
            )
        )

    # ============================================================
    # 9. Copy surface/interface elements
    # ============================================================

    for element in surface_elements:

        new.Add(
            Element2D(
                int(element.index),
                mapped_vertices(element),
            )
        )

    # ============================================================
    # 10. Construct NGSolve mesh
    # ============================================================

    mesh_new = NGSolveMesh(new)
    #mesh_new.ngmesh.Save('Results/' + sweepname + "/Data/deformed.vol")

    # ============================================================
    # 11. Verify vertex numbering and coordinates
    # ============================================================

    max_coordinate_error = 0.0

    old_vertices = list(mesh.vertices)
    new_vertices = list(mesh_new.vertices)

    if len(old_vertices) != len(new_vertices):

        raise RuntimeError(
            "Old and new mesh have different numbers of vertices: "
            f"{len(old_vertices)} vs {len(new_vertices)}"
        )

    for vold, vnew in zip(
        old_vertices,
        new_vertices
    ):

        if vold.nr != vnew.nr:

            raise RuntimeError(
                "Vertex numbering changed: "
                f"old {vold.nr}, new {vnew.nr}"
            )

        pold = mesh[vold].point
        pnew = mesh_new[vnew].point

        d = V(mesh(*pold))

        expected = (
            float(pold[0]) + tau*float(d[0]),
            float(pold[1]) + tau*float(d[1]),
            float(pold[2]) + tau*float(d[2]),
        )

        error = sqrt(
            (float(pnew[0]) - expected[0])**2
            + (float(pnew[1]) - expected[1])**2
            + (float(pnew[2]) - expected[2])**2
        )

        max_coordinate_error = max(
            max_coordinate_error,
            error
        )

    # ============================================================
    # Diagnostics
    # ============================================================

    info = {
        "maximum_displacement":
            maximum_displacement,

        "maximum_displacement_data":
            maximum_displacement_data,

        "maximum_coordinate_error":
            max_coordinate_error,

        "element_type_counts":
            element_counts,

        "descriptor_domains":
            descriptor_domains,
    }

    if verbose:

        print("\nDeformed mesh")
        print("-------------")

        print(
            "Number of vertices:",
            len(new_point_ids)
        )

        print(
            "Tetrahedra:",
            element_counts["tetrahedra"]
        )

        print(
            "Prisms:",
            element_counts["prisms"]
        )

        print(
            "Maximum displacement:",
            maximum_displacement
        )

        print(
            "Maximum coordinate error:",
            max_coordinate_error
        )

        print(
            "Maximum displacement data:",
            maximum_displacement_data
        )

        print(
            "Vertex numbering preserved:",
            True
        )

    print(info)
    from time import sleep
    sleep(10)
    return mesh_new
