import sys
import argparse
from pathlib import Path
import dolfin as d
import numpy as np
from smart import mesh_tools

from phosphorylation_parser_args import add_phosphorylation_preprocess_arguments


def refine(mesh, cell_markers, facet_markers, num_refinements):
    if num_refinements > 0:
        print(
            f"Original mesh has {mesh.num_cells()} cells, "
            f"{mesh.num_facets()} facets and "
            f"{mesh.num_vertices()} vertices"
        )
        d.parameters["refinement_algorithm"] = "plaza_with_parent_facets"

        for _ in range(num_refinements):
            mesh = d.adapt(mesh)
            cell_markers = d.adapt(cell_markers, mesh)
            facet_markers = d.adapt(facet_markers, mesh)

        print(
            f"Refined mesh has {mesh.num_cells()} cells, "
            f"{mesh.num_facets()} facets and "
            f"{mesh.num_vertices()} vertices"
        )

    return mesh, cell_markers, facet_markers


def main(
    mesh_folder: str,
    curRadius: float = 1.0,
    hEdge: float = 0.6,
    num_refinements: int = 0,
    axisymmetric: bool = False,
    rect: bool = False,
):
    print(f"Generating mesh with hEdge={hEdge}")
    # initialize current mesh
    if rect:
        xSize, ySize, zSize = curRadius*10, curRadius*10, curRadius
        cell_mesh = d.BoxMesh(d.Point(0.0, 0.0, 0.0), d.Point(xSize, ySize, zSize), 10, 10, 2**(num_refinements+1))
        cell_markers = d.MeshFunction("size_t", cell_mesh, 3, 1)
        facet_markers = d.MeshFunction("size_t", cell_mesh, 2, 0)
        for f in d.facets(cell_mesh):
            x, y, z = f.midpoint()[:]
            if np.isclose(z, 0.) or np.isclose(z, zSize):
                    # or np.isclose(y, 0.) or np.isclose(y, ySize)\
                    # or np.isclose(z, 0.) or np.isclose(z, zSize):
                facet_markers[f] = 10
    elif axisymmetric:
        cell_mesh, facet_markers, cell_markers = mesh_tools.create_2Dcell(
            outerExpr=f"r**2 + (z-({curRadius}+1))**2 - {curRadius}**2",
            innerExpr="",
            hEdge=hEdge,
        )
        cell_mesh, cell_markers, facet_markers = refine(
            cell_mesh,
            cell_markers,
            facet_markers,
            num_refinements,
        )
        # visualization.plot_dolfin_mesh(cell_mesh, cell_markers)
    else:
        cell_mesh, facet_markers, cell_markers = mesh_tools.create_spheres(
            curRadius, 0, hEdge=hEdge
        )
        cell_mesh, cell_markers, facet_markers = refine(
            cell_mesh,
            cell_markers,
            facet_markers,
            num_refinements,
        )
        # visualization.plot_dolfin_mesh(cell_mesh, cell_markers, facet_markers)

    mesh_folder = Path(mesh_folder)
    mesh_folder.mkdir(exist_ok=True, parents=True)
    mesh_file = mesh_folder / "DemoSphere.h5"
    with d.XDMFFile((mesh_folder / "mesh.xdmf").as_posix()) as f:
        f.write(cell_mesh)
    with d.XDMFFile((mesh_folder / "facet_markers.xdmf").as_posix()) as f:
        f.write(facet_markers)
    with d.XDMFFile((mesh_folder / "cell_markers.xdmf").as_posix()) as f:
        f.write(cell_markers)
    mesh_tools.write_mesh(cell_mesh, facet_markers, cell_markers, mesh_file)
    print("Saved mesh to", mesh_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_phosphorylation_preprocess_arguments(parser)
    args = vars(parser.parse_args())
    raise SystemExit(main(**args))
