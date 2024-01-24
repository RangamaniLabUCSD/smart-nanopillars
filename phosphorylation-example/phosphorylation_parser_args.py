import argparse
from pathlib import Path


def add_phosphorylation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mesh-folder",
        type=Path,
        default=Path.cwd().parent / "meshes-phosphorylation" / "DemoSphere.h5",
    )
    parser.add_argument(
        "-o", "--outdir", type=Path, default=Path("results_phosphorylation")
    )
    parser.add_argument(
        "-dt",
        "--time-step",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--curRadius",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--axisymmetric",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-enforce-mass-conservation",
        action="store_true",
        default=False,
    )


def add_phosphorylation_preprocess_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--mesh-folder",
        type=Path,
        default=Path.cwd().parent / "meshes-phosphorylation",
    )
    parser.add_argument("--curRadius", type=float, default=1.0)
    parser.add_argument("--hEdge", type=float, default=0.2)
    parser.add_argument("--num-refinements", type=int, default=0)
    parser.add_argument(
        "--axisymmetric",
        action="store_true",
        default=False,
    )

def add_phosphorylation_postprocess(
        parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument("results_folder", type=Path)
    parser.add_argument("output_folder", type=Path)