import sys
import json
from textwrap import dedent
from pathlib import Path
import subprocess as sp

tscc_template = dedent(
    """#!/bin/bash
#SBATCH --job-name="{job_name}"
#SBATCH --partition=platinum
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --output=%j-%x-stdout.txt
#SBATCH --error=%j-%x-stderr.txt
#SBATCH --account=csd786
#SBATCH --qos=hcp-csd765
#SBATCH --mem=4G

module load singularitypro/3.11
module load mpich/ge/gcc/64/3.4.2
cd /tscc/nfs/home/eafrancis/gitrepos/smart-nanopillars/scripts

SCRATCH_DIRECTORY=/tscc/lustre/ddn/scratch/eafrancis/nanopillar-sims
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
    script: str,
    submit_tscc: bool,
    dry_run: bool = False,
    job_name: str = "",
):
    in_args = list(map(str, args))
    args_str = " ".join(in_args)
    if dry_run:
        print(f"Run command: {sys.executable} {script} {args_str}")
        return

    if submit_tscc:
        template = tscc_template
    else:
        sp.run([sys.executable, script, *in_args])
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


def preprocess_mech_mesh(
    mesh_folder: Path,
    hEdge: float,
    hInnerEdge: float,
    nanopillar_radius: float,
    nanopillar_height: float,
    nanopillar_spacing: float,
    contact_rad: float,
    nuc_compression: float,
    sym_fraction: float,
    **kwargs,
):
    args = [
        "--mesh-folder",
        Path(mesh_folder).as_posix(),
        "--hEdge",
        hEdge,
        "--hInnerEdge",
        hInnerEdge,
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
        "--sym-fraction",
        sym_fraction,
    ]

    script = (
        (here / ".." / "model-files" / "pre_process_mesh.py")
        .absolute()
        .resolve()
        .as_posix()
    )
    # currently do not generate meshes on cluster, only locally (submit_tscc = False)
    run(
        args=args,
        script=script,
        submit_tscc=False,
    )


def mechanotransduction_example(
    mesh_folder: Path,
    outdir: Path,
    time_step: float,
    e_val: float,
    z_cutoff: float,
    dry_run: bool = False,
    submit_tscc: bool = False,
    curv_sens: float = 0.0,
    reaction_rate_on_np: float = 1.0,
    npc_slope: float = 0.0,
    a0_npc: float = 0.0,
    nuc_compression: float = 0.0,
    WASP_rate: float = 0.0,
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
        "--a0-npc",
        a0_npc,
        "--nuc-compression",
        nuc_compression,
        "--WASP-rate",
        WASP_rate,
    ]

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
        submit_tscc=submit_tscc,
    )

def mechanotransduction_example_nuc_only(
    mesh_folder: Path,
    outdir: Path,
    full_sims_folder: Path,
    time_step: float,
    dry_run: bool = False,
    submit_tscc: bool = False,
    a0_npc: float = 0.0,
    nuc_compression: float = 0.0,
    pore_size: float = 0.5,
    pore_loc: float = 0.0,
    pore_rate: float = 10.0,
    transport_rate: float = 10.0,
    transport_ratio: float = 1.0,
    **kwargs,
):
    args = [
        "--mesh-folder",
        Path(mesh_folder).as_posix(),
        "--full-sims-folder",
        Path(full_sims_folder).as_posix(),
        "--time-step",
        time_step,
        "--a0-npc",
        a0_npc,
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
        submit_tscc=submit_tscc,
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
