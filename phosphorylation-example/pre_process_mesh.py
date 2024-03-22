import sys
import argparse
from pathlib import Path
import dolfin as d
import numpy as np
from smart import mesh_tools

from phosphorylation_parser_args import add_phosphorylation_preprocess_arguments


def refinement2hEdge(refinement: int, hEdge: float = 0.2) -> float:
    return hEdge / 2**refinement


def main(
    mesh_folder: str,
    curRadius: float = 1.0,
    hEdge: float = 0.2,
    num_refinements: int = 0,
    axisymmetric: bool = False,
):
    print(f"Generating mesh with hEdge={hEdge}")
    # initialize current mesh
    if axisymmetric:
        cell_mesh, facet_markers, cell_markers = mesh_tools.create_2Dcell(
            outerExpr=f"r**2 + (z-({curRadius}+1))**2 - {curRadius}**2",
            innerExpr="",
            hEdge=refinement2hEdge(num_refinements, hEdge),
            hInnerEdge=refinement2hEdge(num_refinements, hEdge),
        )
    else:
        cell_mesh, facet_markers, cell_markers = mesh_tools.create_spheres(
            curRadius,
            0,
            hEdge=refinement2hEdge(num_refinements, hEdge),
            hInnerEdge=refinement2hEdge(num_refinements, hEdge),
        )

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
