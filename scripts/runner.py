import sys
from textwrap import dedent
from pathlib import Path
import subprocess as sp

ex3_template = dedent(
    """#!/bin/bash
#SBATCH --job-name="{job_name}"
#SBATCH --partition=fpgaq
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%j-%x-stdout.txt
#SBATCH --error=%j-%x-stderr.txt

module use /cm/shared/ex3-modules/latest/modulefiles
module load  gcc-10.1.0
module load libgfortran-5.0.0
. /home/henriknf/local/src/spack/share/spack/setup-env.sh
spack env activate fenics-dev-fpqga

SCRATCH_DIRECTORY=/global/D1/homes/${{USER}}/smart-comp-sci/{job_name}/${{SLURM_JOBID}}
mkdir -p ${{SCRATCH_DIRECTORY}}
echo "Scratch directory: ${{SCRATCH_DIRECTORY}}"

echo 'Run command: python {script} --outdir "${{SCRATCH_DIRECTORY}}" {args}'
python {script} --outdir "${{SCRATCH_DIRECTORY}}" {args}
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


def run(
    args,
    dry_run: bool,
    script: str,
    submit_ex3: bool,
    submit_saga: bool,
    job_name: str = "",
):
    args = list(map(str, args))
    args_str = " ".join(args)
    if dry_run:
        print(f"Run command: {sys.executable} {script} {args_str}")
        return

    if submit_ex3:
        template = ex3_template
    elif submit_saga:
        template = saga_template
    else:
        sp.run([sys.executable, script, *args])
        return

    job_file = Path("tmp_job.sbatch")
    job_file.write_text(
        template.format(
            job_name=job_name,
            script=script,
            args=args_str,
        )
    )
    sp.run(["sbatch", job_file.as_posix()])
    job_file.unlink()


def preprocess_phosphorylation_mesh(
    mesh_folder: Path,
    curRadius: float,
    hEdge: float,
    num_refinements: int,
    axisymmetric: bool,
    dry_run: bool,
    **kwargs,
):
    args = [
        "--mesh-folder",
        Path(mesh_folder).as_posix(),
        "--curRadius",
        curRadius,
        "--hEdge",
        hEdge,
        "--num-refinements",
        num_refinements,
    ]
    if axisymmetric:
        args.append("--axisymmetric")

    script = (
        (here / ".." / "phosphorylation-example" / "pre_process_mesh.py")
        .absolute()
        .resolve()
        .as_posix()
    )
    run(
        args=args,
        dry_run=dry_run,
        script=script,
        submit_ex3=False,
        submit_saga=False,
    )


def phosphorylation_example(
    mesh_folder: Path,
    outdir: Path,
    time_step: float,
    curRadius: float,
    axisymmetric: bool,
    no_enforce_mass_conservation: bool = False,
    diffusion: float = 10.0,
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
        "--curRadius",
        curRadius,
        "--diffusion",
        diffusion,
    ]
    if axisymmetric:
        args.append("--axisymmetric")

    if no_enforce_mass_conservation:
        args.append("--no-enforce-mass-conservation")

    if submit_ex3 is False and submit_saga is False:
        args.extend(["--outdir", Path(outdir).as_posix()])

    script = (
        (here / ".." / "phosphorylation-example" / "phosphorylation.py")
        .absolute()
        .resolve()
        .as_posix()
    )
    run(
        job_name="phosphorylation",
        args=args,
        dry_run=dry_run,
        script=script,
        submit_ex3=submit_ex3,
        submit_saga=submit_saga,
    )


def postprocess_phosphorylation(
    results_folder: Path,
    output_folder: Path,
    skip_if_processed: bool = False,
    use_tex: bool = False,
    dry_run: bool = False,
    format: str = "png",
    **kwargs,
):
    args = [
        "--results-folder",
        Path(results_folder).as_posix(),
        "--output-folder",
        Path(output_folder).as_posix(),
        "--format",
        format,
    ]
    if skip_if_processed:
        args.append("--skip-if-processed")
    if use_tex:
        args.append("--use-tex")
    script = (
        (here / ".." / "phosphorylation-example" / "postprocess.py")
        .absolute()
        .resolve()
        .as_posix()
    )
    run(
        args=args,
        dry_run=dry_run,
        script=script,
        submit_ex3=False,
        submit_saga=False,
    )


def preprocess_mito_mesh(
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
    run(
        args=args,
        dry_run=dry_run,
        script=script,
        submit_ex3=False,
        submit_saga=False,
    )


def mito_example(
    mesh_file: Path,
    curv_file: Path,
    outdir: Path,
    time_step: float,
    curv_dep: float,
    enforce_mass_conservation: bool,
    D: float,
    dry_run: bool = False,
    submit_ex3: bool = False,
    submit_saga: bool = False,
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
        (here / ".."/ ".." / "mito-example" / "mito_atp_dynamics.py")
        .absolute()
        .resolve()
        .as_posix()
    )
    run(
        job_name="mito",
        args=args,
        dry_run=dry_run,
        script=script,
        submit_ex3=submit_ex3,
        submit_saga=submit_saga,
    )


def pre_preprocess_mech_mesh(
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
    run(
        args=args,
        dry_run=dry_run,
        script=script,
        submit_ex3=False,
        submit_saga=False,
    )


def mechanotransduction_example(
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
    ]
    if axisymmetric:
        args.append("--axisymmetric")
    if well_mixed:
        args.append("--well-mixed")
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
    run(
        job_name="mechanotransduction",
        args=args,
        dry_run=dry_run,
        script=script,
        submit_ex3=submit_ex3,
        submit_saga=submit_saga,
    )


def postprocess_mechanotransduction(
    results_folder: Path, output_folder: Path, dry_run: bool = False, **kwargs
):
    args = [
        "--results-folder",
        Path(results_folder).as_posix(),
        "--output-folder",
        Path(output_folder).as_posix(),
    ]
    script = (
        (here / ".." / "mechanotransduction-example" / "postprocess.py")
        .absolute()
        .resolve()
        .as_posix()
    )
    run(
        args=args,
        dry_run=dry_run,
        script=script,
        submit_ex3=False,
        submit_saga=False,
    )


def preprocess_cru_mesh(
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
        (here / ".." / "ca2+-examples" / "pre_process_mesh_cru.py")
        .absolute()
        .resolve()
        .as_posix()
    )
    run(
        args=args,
        dry_run=dry_run,
        script=script,
        submit_ex3=False,
        submit_saga=False,
    )


def cru_example(
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
        (here / ".." / "ca2+-examples" / "cru.py")
        .absolute()
        .resolve()
        .as_posix()
    )
    run(
        job_name="cru",
        args=args,
        dry_run=dry_run,
        script=script,
        submit_ex3=submit_ex3,
        submit_saga=submit_saga,
    )


def preprocess_spine_mesh(
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
    run(
        args=args,
        dry_run=dry_run,
        script=script,
        submit_ex3=False,
        submit_saga=False,
    )


def dendritic_spine_example(
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
    run(
        job_name="dendritic_spine",
        args=args,
        dry_run=dry_run,
        script=script,
        submit_ex3=submit_ex3,
        submit_saga=submit_saga,
    )


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
