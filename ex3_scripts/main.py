import sys
from textwrap import dedent
from pathlib import Path
import argparse
import subprocess as sp

ex3_template = dedent(
    """#!/bin/bash
#SBATCH --job-name="{job_name}"
#SBATCH --partition=defq
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%j-%x-stdout.txt
#SBATCH --error=%j-%x-stderr.txt


module use /cm/shared/ex3-modules/202309a/defq/modulefiles
module load python-fenics-dolfin-2019.2.0.dev0

SCRATCH_DIRECTORY=/global/D1/homes/${{USER}}/smart-comp-sci/{job_name}/${{SLURM_JOBID}}
mkdir -p ${{SCRATCH_DIRECTORY}}
echo "Scratch directory: ${{SCRATCH_DIRECTORY}}"

echo 'Run command: {python} {script} --outdir "${{SCRATCH_DIRECTORY}}" {args}'
{python} {script} --outdir "${{SCRATCH_DIRECTORY}}" {args}
# Move log file to results folder
mv ${{SLURM_JOBID}}-* ${{SCRATCH_DIRECTORY}}
"""
)

saga_template = dedent(
    """#!/bin/bash
#SBATCH --job-name="{job_name}"
#SBATCH --account=nn9249k
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%j-%x-stdout.txt
#SBATCH --error=%j-%x-stderr.txt


source /cluster/shared/fenics/conf/fenics-2019.2.0.dev0-2023-05-24.saga.foss-2022a-py3.10.conf

SUBMIT_DIRECTORY=/cluster/home/henriknf/local/src/smart-comp-sci/ex3_scripts
SCRATCH_DIRECTORY=${{SUBMIT_DIRECTORY}}/results/{job_name}/${{SLURM_JOBID}}
mkdir -p ${{SCRATCH_DIRECTORY}}
echo "Scratch directory: ${{SCRATCH_DIRECTORY}}"

echo 'Run command: python3 {script} --outdir "${{SCRATCH_DIRECTORY}}" {args}'
srun python3 {script} --outdir "${{SCRATCH_DIRECTORY}}" {args}
# Move log file to results folder
mv ${{SUBMIT_DIRECTORY}}/${{SLURM_JOBID}}-* ${{SCRATCH_DIRECTORY}}
"""
)


here = Path(__file__).parent.absolute()


def add_argument_dendritic_spine_example(parser: argparse.ArgumentParser):
    sys.path.insert(0, (here / ".." / "ca2+-examples").as_posix())
    import ca2_parser_args

    ca2_parser_args.add_run_dendritic_spine_arguments(parser)


def add_argument_preprocess_spine_mesh(parser: argparse.ArgumentParser):
    sys.path.insert(0, (here / ".." / "ca2+-examples").as_posix())
    import ca2_parser_args

    ca2_parser_args.add_preprocess_spine_mesh_arguments(parser)


def add_argument_mechanotransduction_example(parser: argparse.ArgumentParser):
    sys.path.insert(0, (here / ".." / "mechanotransduction-example").as_posix())
    import mech_parser_args

    mech_parser_args.add_mechanotransduction_arguments(parser)


def add_argument_preprocess_mech_mesh(parser: argparse.ArgumentParser):
    sys.path.insert(0, (here / ".." / "mechanotransduction-example").as_posix())
    import mech_parser_args

    mech_parser_args.add_preprocess_mech_mesh_arguments(parser)

def add_argument_mito_example(parser: argparse.ArgumentParser):
    sys.path.insert(0, (here / ".." / "mito-example").as_posix())
    import mito_parser_args

    mito_parser_args.add_run_mito_arguments(parser)


def add_argument_preprocess_mito_mesh(parser: argparse.ArgumentParser):
    sys.path.insert(0, (here / ".." / "mito-example").as_posix())
    import mito_parser_args

    mito_parser_args.add_preprocess_mito_mesh_arguments(parser)

def run_preprocess_mito_mesh(
    input_mesh_file: Path,
    output_mesh_file: Path,
    input_curv_file: Path,
    output_curv_file: Path,
    dry_run: bool,
    num_refinements: int,
    single_compartment_im: bool,
    **kwargs,
):
    args = [
        "--input-mesh-file",
        Path(input_mesh_file).as_posix(),
        "--output-mesh-file",
        Path(output_mesh_file).as_posix(),
        "--input-curv-file",
        Path(input_curv_file).as_posix(),
        "--output-curv-file",
        Path(output_curv_file).as_posix(),
        "--num-refinements",
        num_refinements,
    ]

    if single_compartment_im:
        args.append("--single-compartment-im")

    script = (
        (here / ".." / "mito-example" / "pre_process_mesh.py")
        .absolute()
        .resolve()
        .as_posix()
    )
    args = list(map(str, args))
    if dry_run:
        print(f"Run command: {sys.executable} {script} {' '.join(args)}")
        return

    sp.run([sys.executable, script, *args])


def run_mito_example(
    mesh_file: Path,
    curv_file: Path,
    outdir: Path,
    time_step: float,
    curv_dep: float,
    enforce_mass_conservation: bool,
    D: float,
    dry_run: bool = False,
    submit_ex3: bool = False,
    **kwargs,
):
    args = [
        "--mesh-file",
        Path(mesh_file).as_posix(),
        "--curv-file",
        Path(curv_file).as_posix(),
        "--time-step",
        time_step,
        "--curv-dep",
        curv_dep,
        "--D",
        D,
    ]
    if enforce_mass_conservation:
        args.append("--enforce-mass-conservation")

    if not submit_ex3:
        args.extend(["--outdir", Path(outdir).as_posix()])

    script = (
        (here / ".." / "mito-example" / "mito_atp_dynamics.py")
        .absolute()
        .resolve()
        .as_posix()
    )
    # Turn all arguments into strings
    args = list(map(str, args))
    args_str = " ".join(args)
    if dry_run:
        print(f"Run command: {sys.executable} {script} {args_str}")
        return

    if submit_ex3:
        job_file = Path("tmp_job.sbatch")
        job_file.write_text(
            ex3_template.format(
                job_name="mito",
                python=sys.executable,
                script=script,
                args=args_str,
            )
        )

        sp.run(["sbatch", job_file])
        job_file.unlink()
    else:
        sp.run([sys.executable, script, *args])


def run_pre_preprocess_mech_mesh(
    mesh_folder: Path,
    shape: str,
    hEdge: float,
    hInnerEdge: float,
    num_refinements: int,
    dry_run: bool,
    **kwargs,
):
    args = [
        "--mesh-folder",
        Path(mesh_folder).as_posix(),
        "--shape",
        shape,
        "--hEdge",
        hEdge,
        "--hInnerEdge",
        hInnerEdge,
        "--num-refinements",
        num_refinements,
    ]

    script = (
        (here / ".." / "mechanotransduction-example" / "pre_process_mesh.py")
        .absolute()
        .resolve()
        .as_posix()
    )
    args = list(map(str, args))
    print(f"Run command: {sys.executable} {script} {' '.join(args)}")
    if dry_run:
        return
    sp.run([sys.executable, script, *args])


def run_mechanotransduction_example(
    mesh_folder: Path,
    outdir: Path,
    time_step: float,
    e_val: float,
    z_cutoff: float,
    axisymmetric: bool,
    well_mixed: bool,
    no_enforce_mass_conservation: bool = False,
    dry_run: bool = False,
    submit_ex3: bool = False,
    submit_saga: bool = False,
    **kwargs,
):
    args = [
        "--mesh-folder",
        Path(mesh_folder).as_posix(),
        "--time-step",
        time_step,
        "--e-val",
        e_val,
        "--z-cutoff",
        z_cutoff,
        "--axisymmetric" if axisymmetric else "",
        "--well-mixed" if well_mixed else "",
    ]
    if no_enforce_mass_conservation:
        args.append("--no-enforce-mass-conservation")

    if submit_ex3 is False and submit_saga is False:
        args.extend(["--outdir", Path(outdir).as_posix()])
    

    script = (
        (here / ".." / "mechanotransduction-example" / "mechanotransduction.py")
        .absolute()
        .resolve()
        .as_posix()
    )
    args = list(map(str, args))
    args_str = " ".join(args)
    if dry_run:
        print(f"Run command: {sys.executable} {script} {args_str}")
        return

    if submit_ex3:
        job_file = Path("tmp_job.sbatch")
        job_file.write_text(
            ex3_template.format(
                job_name="mechanotransduction",
                python=sys.executable,
                script=script,
                args=args_str,
            )
        )

        sp.run(["sbatch", job_file])
        job_file.unlink()
    elif submit_saga:
        job_file = Path("tmp_job.sbatch")
        job_file.write_text(
            saga_template.format(
                job_name="mechanotransduction",
                script=script,
                args=args_str,
            )
        )

        sp.run(["sbatch", job_file])
        job_file.unlink()

    else:
        sp.run([sys.executable, script, *args])


def run_preprocess_spine_mesh(
    input_mesh_file: Path,
    output_mesh_file: Path,
    dry_run: bool,
    num_refinements: int,
    **kwargs,
):
    args = [
        "--input-mesh-file",
        Path(input_mesh_file).as_posix(),
        "--output-mesh-file",
        Path(output_mesh_file).as_posix(),
        "--num-refinements",
        num_refinements,
    ]

    script = (
        (here / ".." / "ca2+-examples" / "pre_process_mesh.py")
        .absolute()
        .resolve()
        .as_posix()
    )
    args = list(map(str, args))
    if dry_run:
        print(f"Run command: {sys.executable} {script} {' '.join(args)}")
        return

    sp.run([sys.executable, script, *args])


def run_dendritic_spine_example(
    mesh_file: Path,
    outdir: Path,
    time_step: float,
    enforce_mass_conservation: bool,
    dry_run: bool = False,
    submit_ex3: bool = False,
    submit_saga: bool = False,
    **kwargs,
):
    args = [
        "--mesh-file",
        Path(mesh_file).as_posix(),
        "--time-step",
        time_step,
    ]
    if enforce_mass_conservation:
        args.append("--enforce-mass-conservation")

    if submit_ex3 is False and submit_saga is False:
        args.extend(["--outdir", Path(outdir).as_posix()])

    script = (
        (here / ".." / "ca2+-examples" / "dendritic_spine.py")
        .absolute()
        .resolve()
        .as_posix()
    )
    # Turn all arguments into strings
    args = list(map(str, args))
    args_str = " ".join(args)
    if dry_run:
        print(f"Run command: {sys.executable} {script} {args_str}")
        return

    if submit_ex3:
        job_file = Path("tmp_job.sbatch")
        job_file.write_text(
            ex3_template.format(
                job_name="dendritic-spine",
                python=sys.executable,
                script=script,
                args=args_str,
            )
        )

        sp.run(["sbatch", job_file])
        job_file.unlink()
    elif submit_saga:
        job_file = Path("tmp_job.sbatch")
        job_file.write_text(
            saga_template.format(
                job_name="dendritic-spine",
                script=script,
                args=args_str,
            )
        )

        sp.run(["sbatch", job_file])
        job_file.unlink()
    else:
        sp.run([sys.executable, script, *args])


def convert_notebooks(dry_run: bool = False, **kwargs):
    import jupytext

    for example_folder in (here / "..").iterdir():
        # Only look in folders that contains the word example
        if "example" not in example_folder.stem:
            continue

        # Loop over all files in that folder
        for f in example_folder.iterdir():
            if not f.suffix == ".ipynb":
                continue

            print(f"Convert {f} to {f.with_suffix('.py')}")
            if dry_run:
                continue

            text = jupytext.reads(f.read_text())
            jupytext.write(text, f.with_suffix(".py"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just print the command and do not run it",
    )
    parser.add_argument(
        "--submit-ex3",
        action="store_true",
        help="Add this flag if you want to submit the job on the ex3 cluster",
    )
    parser.add_argument(
        "--submit-saga",
        action="store_true",
        help="Add this flag if you want to submit the job on the saga cluster",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("convert-notebooks", help="Convert notebooks to python files")

    # Dendritic spine
    preprocess_spine_mesh = subparsers.add_parser(
        "preprocess-spine-mesh", help="Preprocess mesh for dendritic spine example"
    )
    add_argument_preprocess_spine_mesh(preprocess_spine_mesh)

    dendritic_spine = subparsers.add_parser(
        "dendritic-spine", help="Run dendritic spine example"
    )
    add_argument_dendritic_spine_example(dendritic_spine)

    # Mechanotransduction example
    preprocess_spine_mesh = subparsers.add_parser(
        "preprocess-mech-mesh", help="Preprocess mesh for mechanotransduction example"
    )
    add_argument_preprocess_mech_mesh(preprocess_spine_mesh)

    dendritic_spine = subparsers.add_parser(
        "mechanotransduction", help="Run mechanotransduction example"
    )
    add_argument_mechanotransduction_example(dendritic_spine)

    # Mito example
    preprocess_spine_mesh = subparsers.add_parser(
        "preprocess-mito-mesh", help="Preprocess mesh for mito example"
    )
    add_argument_preprocess_mito_mesh(preprocess_spine_mesh)

    mito = subparsers.add_parser(
        "mito", help="Run mito example"
    )
    add_argument_mito_example(mito)

    args = vars(parser.parse_args())
    # if args[""]

    if args["command"] == "convert-notebooks":
        print("Convert notebook")
        convert_notebooks(**args)

    elif args["command"] == "preprocess-spine-mesh":
        print("Run preprocess dendritic spine mesh")
        run_preprocess_spine_mesh(**args)

    elif args["command"] == "dendritic-spine":
        print("Run dendritic spine example")
        run_dendritic_spine_example(**args)

    elif args["command"] == "preprocess-mech-mesh":
        print("Run preprocess mechanotransduction mesh")
        run_pre_preprocess_mech_mesh(**args)

    elif args["command"] == "mechanotransduction":
        print("Run mechanotransduction example")
        run_mechanotransduction_example(**args)
    
    elif args["command"] == "preprocess-mito-mesh":
        print("Run preprocess mito mesh")
        run_preprocess_mito_mesh(**args)

    elif args["command"] == "mito":
        print("Run mito example")
        run_mito_example(**args)


if __name__ == "__main__":
    raise SystemExit(main())
