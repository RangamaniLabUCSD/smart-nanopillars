import argparse
from pathlib import Path

def add_mechanotransduction_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mesh-folder",
        type=Path,
        default=Path.cwd().parent / "meshes_local" / "spreadCell_mesh_circle_R13smooth",
    )
    parser.add_argument(
        "-o", "--outdir", type=Path, default=Path("results_mechanotransduction")
    )
    parser.add_argument(
        "-dt",
        "--time-step",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--e-val",
        type=float,
        default=7e7,
    ) # stiffness for glass coverslip
    parser.add_argument(
        "--z-cutoff",
        type=float,
        default=1e-4,
    ) # stimulation only below z-cutoff
    parser.add_argument(
        "--curv-sens",
        type=float,
        default=0.0,
    ) # curvature sensitivity factor of FAK phosph.
    parser.add_argument(
        "--reaction-rate-on-np",
        type=float,
        default=1.0,
    ) # fractional FAK phosph. rate on nanopillars
    parser.add_argument(
        "--nuc-compression",
        type=float,
        default=0.0,
    ) # nuc indentation on nanopillars
    parser.add_argument(
        "--axisymmetric",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--well-mixed",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--npc-slope",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--a0-npc",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--WASP-rate",
        type=float,
        default=0.0,
    )

def add_mechanotransduction_nucOnly_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mesh-folder",
        type=Path,
        default=Path.cwd().parent / "mesh",
    )
    parser.add_argument(
        "--full-sims-folder",
        type=Path,
        default=Path.cwd().parent / "results",
    )
    parser.add_argument(
        "-o", "--outdir", type=Path, default=Path("results_nucOnly")
    )
    parser.add_argument(
        "-dt",
        "--time-step",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--a0-npc",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--nuc-compression",
        type=float,
        default=0.0,
    ) # nuc indentation on nanopillars
    parser.add_argument(
        "--pore-size",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--pore-loc",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--pore-rate",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--transport-rate",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--transport-ratio",
        type=float,
        default=1.0,
    )


def add_preprocess_mech_mesh_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mesh-folder",
        type=Path,
        default=Path.cwd().parent / "meshes_local" / "spreadCell_mesh_circle_R13smooth",
    )
    parser.add_argument(
        "--shape", type=str, default="circle", choices=Shape._member_names_
    )
    parser.add_argument("--hEdge", type=float, default=0.6)
    parser.add_argument("--hInnerEdge", type=float, default=0.6)
    parser.add_argument("--nanopillar-radius", type=float, default=0.0)
    parser.add_argument("--nanopillar-height", type=float, default=0.0)
    parser.add_argument("--nanopillar-spacing", type=float, default=0.0)
    parser.add_argument("--contact-rad", type=float, default=13.0)
    parser.add_argument("--nuc-compression", type=float, default=0.0)
    parser.add_argument("--sym-fraction", type=float, default=1/8)