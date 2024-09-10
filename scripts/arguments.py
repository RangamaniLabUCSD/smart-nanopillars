import sys
from pathlib import Path
import argparse


here = Path(__file__).parent.absolute()


def mechanotransduction_example(parser: argparse.ArgumentParser):
    sys.path.insert(0, (here / ".." / "model-files").as_posix())
    import mech_parser_args

    mech_parser_args.add_mechanotransduction_arguments(parser)

def mechanotransduction_nucOnly_example(parser: argparse.ArgumentParser):
    sys.path.insert(0, (here / ".." / "model-files").as_posix())
    import mech_parser_args

    mech_parser_args.add_mechanotransduction_nucOnly_arguments(parser)


def mechanotransduction_postprocess(parser: argparse.ArgumentParser):
    sys.path.insert(0, (here / ".." / "model-files").as_posix())
    import mech_parser_args

    mech_parser_args.add_mechanotransduction_postprocess_arguments(parser)


def preprocess_mech_mesh(parser: argparse.ArgumentParser):
    sys.path.insert(0, (here / ".." / "model-files").as_posix())
    import mech_parser_args

    mech_parser_args.add_preprocess_mech_mesh_arguments(parser)



def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Root parser
    parser.add_argument(
        "-n",
        "--ntasks",
        default=1,
        type=int,
        help="Number of cores to use for each program",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just print the command and do not run it",
    )


    cg = parser.add_argument_group("cluster", "Options relevant for clusters")
    cg.add_argument(
        "-p",
        "--partition",
        default="defq",
        type=str,
        help="Which partition to use on the cluster",
    )
    cluster = cg.add_mutually_exclusive_group()
    cluster.add_argument(
        "--submit-ex3",
        action="store_true",
        help="Add this flag if you want to submit the job on the ex3 cluster",
    )
    cluster.add_argument(
        "--submit-tscc",
        action="store_true",
        help="Add this flag if you want to submit the job on the hopper cluster at TSCC",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Convert notebooks
    subparsers.add_parser("convert-notebooks", help="Convert notebooks to python files")

    # Mechanotransduction example
    preprocess_mech_mesh_parser = subparsers.add_parser(
        "mechanotransduction-preprocess", help="Preprocess mesh for mechanotransduction example"
    )
    preprocess_mech_mesh(preprocess_mech_mesh_parser)

    mechanotransduction_parser = subparsers.add_parser(
        "mechanotransduction", help="Run mechanotransduction example"
    )
    mechanotransduction_example(mechanotransduction_parser)

    mechanotransduction_nucOnly_parser = subparsers.add_parser(
        "mechanotransduction-nuc-only", help="Run mechanotransduction example with pore formation"
    )
    mechanotransduction_nucOnly_example(mechanotransduction_nucOnly_parser)

    mechanotransduction_postprocess_parser = subparsers.add_parser(
        "mechanotransduction-postprocess", help="Postprocess mechanotransduction example"
    )
    mechanotransduction_postprocess(mechanotransduction_postprocess_parser)
    return parser