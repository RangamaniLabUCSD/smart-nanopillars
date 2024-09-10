import sys
import json
from textwrap import dedent
from pathlib import Path
import subprocess as sp

# could specify memory below in SLURM script as #SBATCH --mem=4G
tscc_template = dedent(
    """#!/bin/bash
#SBATCH --job-name="{job_name}"
#SBATCH --partition=platinum
#SBATCH --time=200:00:00
#SBATCH --ntasks={ntasks}
#SBATCH --output=%j-%x-stdout.txt
#SBATCH --error=%j-%x-stderr.txt
#SBATCH --account=csd786
#SBATCH --qos=hcp-csd765
#SBATCH --mem=4G

module load singularitypro/3.11
module load mpich/ge/gcc/64/3.4.2
cd /tscc/nfs/home/eafrancis/gitrepos/smart-comp-sci/scripts

SCRATCH_DIRECTORY=/tscc/lustre/ddn/scratch/eafrancis/smart-comp-sci-data/{job_name}
mkdir -p ${{SCRATCH_DIRECTORY}}
echo "Scratch directory: ${{SCRATCH_DIRECTORY}}"

echo 'Run command in container: python {script} {args}'
singularity exec --bind $HOME:/root/shared,\
$TMPDIR:/root/tmp,$SCRATCH_DIRECTORY:/root/scratch \
/tscc/nfs/home/eafrancis/smart-newmeshview.sif \
python3 {script} {args}

# Move log file to results folder
mv ${{SLURM_JOBID}}-* ${{SCRATCH_DIRECTORY}}
"""
)

here = Path(__file__).parent.absolute()


def run(
    args,
    dry_run: bool,
    script: str,
    submit_ex3: bool,  # FIXME: Change this and next param to enum
    submit_tscc: bool,
    job_name: str = "",
    ntasks: int = 1,
    partition: str = "defq",
):
    in_args = list(map(str, args))
    args_str = " ".join(in_args)
    if dry_run:
        print(f"Run command: {sys.executable} {script} {args_str}")
        return

    if submit_tscc:
        template = tscc_template
    else:
        sp.run(["mpirun", "-n", str(ntasks), sys.executable, script, *in_args])
        return

    job_file = Path("tmp_job.sbatch")
    job_file.write_text(
        template.format(
            job_name=job_name,
            script=script,
            args=args_str,
            ntasks=ntasks,
            partition=partition,
        )
    )
    sp.run(["sbatch", job_file.as_posix()])
    job_file.unlink()


def pre_preprocess_mech_mesh(
    mesh_folder: Path,
    shape: str,
    hEdge: float,
    hInnerEdge: float,
    num_refinements: int,
    dry_run: bool,
    full_3d: bool,
    nanopillar_radius: float,
    nanopillar_height: float,
    nanopillar_spacing: float,
    contact_rad: float,
    nuc_compression: float,
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
        "--nanopillar-radius",
        nanopillar_radius,
        "--nanopillar-height",
        nanopillar_height,
        "--nanopillar-spacing",
        nanopillar_spacing,
        "--contact-rad",
        contact_rad,
        "--nuc-compression",
        nuc_compression,
    ]

    if full_3d:
        args.append("--full-3d")

    script = (
        (here / ".." / "model-files" / "pre_process_mesh.py")
        .absolute()
        .resolve()
        .as_posix()
    )
    run(
        args=args,
        dry_run=dry_run,
        script=script,
        submit_ex3=False,
        submit_tscc=False,
    )


def mechanotransduction_example(
    mesh_folder: Path,
    outdir: Path,
    time_step: float,
    e_val: float,
    z_cutoff: float,
    axisymmetric: bool,
    well_mixed: bool,
    dry_run: bool = False,
    submit_tscc: bool = False,
    ntasks: int = 1,
    partition: str = "defq",
    curv_sens: float = 0.0,
    reaction_rate_on_np = 1.0,
    npc_slope = 0.0,
    u0_npc = 0.0,
    nuc_compression = 0.0,
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
        "--curv-sens",
        curv_sens,
        "--reaction-rate-on-np",
        reaction_rate_on_np,
        "--npc-slope",
        npc_slope,
        "--u0-npc",
        u0_npc,
        "--nuc-compression",
        nuc_compression,
    ]
    if axisymmetric:
        args.append("--axisymmetric")
    if well_mixed:
        args.append("--well-mixed")

    args.extend(["--outdir", Path(outdir).as_posix()])

    script = (
        (here / ".." / "model-files" / "mechanotransduction.py")
        .absolute()
        .resolve()
        .as_posix()
    )
    run(
        job_name="mechanotransduction",
        args=args,
        dry_run=dry_run,
        script=script,
        submit_ex3=False,
        submit_tscc=submit_tscc,
        ntasks=ntasks,
        partition=partition,
    )

def mechanotransduction_example_nuc_only(
    mesh_folder: Path,
    outdir: Path,
    full_sims_folder: Path,
    time_step: float,
    dry_run: bool = False,
    submit_tscc: bool = False,
    ntasks: int = 1,
    partition: str = "defq",
    u0_npc = 0.0,
    nuc_compression = 0.0,
    pore_size = 0.5,
    pore_loc = 0.0,
    pore_rate = 10.0,
    transport_rate = 10.0,
    transport_ratio = 1.0,
    **kwargs,
):
    args = [
        "--mesh-folder",
        Path(mesh_folder).as_posix(),
        "--full-sims-folder",
        Path(full_sims_folder).as_posix(),
        "--time-step",
        time_step,
        "--u0-npc",
        u0_npc,
        "--nuc-compression",
        nuc_compression,
        "--pore-size",
        pore_size,
        "--pore-loc",
        pore_loc,
        "--pore-rate",
        pore_rate,
        "--transport-rate",
        transport_rate,
        "--transport-ratio",
        transport_ratio
    ]

    args.extend(["--outdir", Path(outdir).as_posix()])

    script = (
        (here / ".." / "model-files" / "mechanotransduction_nucTransportOnly.py")
        .absolute()
        .resolve()
        .as_posix()
    )
    run(
        job_name="mechanotransduction",
        args=args,
        dry_run=dry_run,
        script=script,
        submit_ex3=False,
        submit_tscc=submit_tscc,
        ntasks=ntasks,
        partition=partition,
    )


def postprocess_mechanotransduction(
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
        (here / ".." / "model-files" / "postprocess.py")
        .absolute()
        .resolve()
        .as_posix()
    )
    run(
        args=args,
        dry_run=dry_run,
        script=script,
        submit_ex3=False,
        submit_tscc=False,
    )


def convert_notebooks(dry_run: bool = False, **kwargs):
    import jupytext

    for example_folder in (here / "..").iterdir():
        # Only look in folders that contains the word model
        if "model" not in example_folder.stem:
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
