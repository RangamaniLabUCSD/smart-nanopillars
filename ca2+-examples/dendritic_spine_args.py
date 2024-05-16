import argparse
from pathlib import Path

def add_run_dendritic_spine_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mesh-file",
        type=Path,
        default=Path.cwd() / "spine_mesh" /  "spine_mesh.h5"
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=Path("results_dendritic_spine")
    )
    parser.add_argument(
        "-dt",
        "--time-step",
        type=float,
        default=0.0002,
    )
    parser.add_argument(
        "--enforce-mass-conservation",
        action='store_true'
    )


def add_preprocess_spine_mesh_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-mesh-file",
        type=Path,
        default=Path.cwd().parent / "meshes_local" / "1spine_PM10_PSD11_ERM12_cyto1_ER2_coarser.xml"
    )
    parser.add_argument(
        "--output-mesh-file",
        type=Path,
        default=Path("spine_mesh.h5")
    )
    parser.add_argument(
        "-n",
        "--num-refinements",
        type=int,
        default=0,
    )