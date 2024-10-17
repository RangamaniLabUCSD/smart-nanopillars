import sys
import argparse
from pathlib import Path
import dolfin as d
import numpy as np
from smart import mesh_tools

from mech_parser_args import (
    add_preprocess_mech_mesh_arguments,
    Shape,
    shape2symfraction,
)


# Define expressions to be used for geometry definitions.


outerExpr15 = "(1 - z**4/(1000+z**4)) * (r**2 + z**2) + 0.4*(r**2 + (z+15)**2)*z**4 / (10 + z**4) - 225"
innerExpr15 = "(r/5.3)**2 + ((z-7.8/2)/2.4)**2 - 1"
outerExpr13 = "(1 - z**4/(2000+z**4)) * (r**2 + z**2) + 0.4*(r**2 + (z+9.72)**2)*z**4 / (15 + z**4) - 169"
innerExpr13 = "(r/5.3)**2 + ((z-9.6/2)/2.4)**2 - 1"
outerExpr10 = "(1 - z**4/(35+z**4)) * (r**2 + z**2) + 0.94*(r**2 + (z+.01)**2)*z**4 / (30 + z**4) - 100"
innerExpr10 = "(r/5.3)**2 + ((z-5)/2.4)**2 - 1"
sphCapOuter = "r**2 + z**2 - 100"
sphCapInner = "r**2 + (z-4)**2 - 9"


def shape2theta(shape: Shape) -> str:
    # Based on thetaStr = ["", "", "", star_str1, star_str1, star_str1, "rect0.6", "rect0.6", "rect0.6"]
    # Other values are
    # star_str1 = "0.98 + 0.2814*cos(5*theta)"
    # star_str2 = "0.95 + 0.4416*cos(5*theta)"
    # star_str2 = "0.9 + 0.6164*cos(5*theta)"
    if Shape[shape].value == "circle":
        return ""
    elif Shape[shape].value == "star":
        return "0.98 + 0.2814*cos(5*theta)"
    elif Shape[shape].value == "rect":
        return "rect0.6"
    else:
        raise ValueError(f"Invalid shape {shape}")


def refine(mesh, cell_markers, facet_markers, num_refinements, curv_markers=None):
    if num_refinements > 0:
        print(
            f"Original mesh has {mesh.num_cells()} cells, "
            f"{mesh.num_facets()} facets and "
            f"{mesh.num_vertices()} vertices"
        )
        d.parameters["refinement_algorithm"] = "plaza_with_parent_facets"
        if curv_markers is not None:
            f_orig = d.Function(d.FunctionSpace(mesh, "CG", 1))
            f_orig.set_allow_extrapolation(True)
            f_orig.vector()[:] = curv_markers.array()
        for _ in range(num_refinements):
            mesh = d.adapt(mesh)
            cell_markers = d.adapt(cell_markers, mesh)
            facet_markers = d.adapt(facet_markers, mesh)

        print(
            f"Refined mesh has {mesh.num_cells()} cells, "
            f"{mesh.num_facets()} facets and "
            f"{mesh.num_vertices()} vertices"
        )
        if curv_markers is not None:
            adapted_curv_markers = d.MeshFunction("double", mesh, 0)
            f = d.interpolate(f_orig, d.FunctionSpace(mesh, "CG", 1))
            adapted_curv_markers.array()[:] = f.vector()[:]
            return mesh, cell_markers, facet_markers, adapted_curv_markers

    return mesh, cell_markers, facet_markers, curv_markers


def main(
    mesh_folder: str,
    shape: Shape,
    hEdge: float = 0.6,
    hInnerEdge: float = 0.6,
    num_refinements: int = 0,
    full_3d = False,
    nanopillar_radius: float = 0,
    nanopillar_height: float = 0,
    nanopillar_spacing: float = 0,
    contact_rad: float = 13.0,
    nuc_compression: float = 0.0,
):
    here = Path(__file__).parent.absolute()
    sys.path.append((here / ".." / "utils").as_posix())

    import spread_cell_mesh_generation as mesh_gen

    hNP = hEdge * 0.3
    nanopillars = [nanopillar_radius, nanopillar_height, nanopillar_spacing]
    cell_mesh, facet_markers, cell_markers, substrate_markers, curv_markers, u_nuc, a_nuc = mesh_gen.create_3dcell(
                                                                        contactRad=contact_rad,
                                                                        hEdge=hEdge, hInnerEdge=hInnerEdge,
                                                                        hNP=hNP,
                                                                        thetaExpr=shape2theta(shape),
                                                                        nanopillars=nanopillars,
                                                                        return_curvature=True,
                                                                        sym_fraction=1/8,
                                                                        nuc_compression=nuc_compression)

    mesh_folder = Path(mesh_folder)
    mesh_folder.mkdir(exist_ok=True, parents=True)
    mesh_file = mesh_folder / "spreadCell_mesh.h5"
    mesh_tools.write_mesh(cell_mesh, facet_markers, cell_markers, mesh_file, [substrate_markers])
    d.File(str(mesh_folder / "facets.pvd")) << facet_markers
    d.File(str(mesh_folder / "cells.pvd")) << cell_markers
    # save curvatures for reference
    curv_file_name = mesh_folder / "curvatures.xdmf"
    with d.XDMFFile(str(curv_file_name)) as curv_file:
        curv_file.write(curv_markers)
    # save nuclear deformations if applicable
    # if nuc_compression > 0:
    #     u_nuc_file = d.XDMFFile(str(mesh_folder / "u_nuc.xdmf"))
    #     u_nuc_file.write_checkpoint(u_nuc, "u_nuc", 0)#, d.XDMFFile.Encoding.HDF5, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_preprocess_mech_mesh_arguments(parser)
    args = vars(parser.parse_args())
    raise SystemExit(main(**args))