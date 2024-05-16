from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import json
from typing import NamedTuple, Any
from itertools import cycle
import numpy as np
import re
import matplotlib.pyplot as plt

import dendritic_spine_args


ntasks_pattern = re.compile("ntasks: (?P<n>\d+)")


class Data(NamedTuple):
    timings_: list[dict[str, Any]]
    config: dict[str, Any]
    t: np.ndarray
    concVec: np.ndarray
    gradVec: np.ndarray
    stderr: str
    stdout: str
    ntasks: int
    folder: Path = None

    @property
    def mesh(self) -> str:
        return Path(self.config["mesh_file"]).stem

    @property
    def dt(self) -> float:
        return self.config["solver"]["initial_dt"]

    @property
    def timings(self):
        return pd.DataFrame(self.timings_)

    @property
    def num_refinements(self) -> int:
        if "refined" in self.mesh:
            return int(self.mesh.split("_")[-1])
        return 0

    @property
    def total_run_time(self) -> float:
        try:
            return self.timings[self.timings["name"] == "dendritic-spine-example"][
                "wall tot"
            ].values[0]
        except IndexError:
            return np.nan

    def to_json(self) -> dict[str, Any]:
        return {
            "t": self.t.tolist(),
            "concVec": self.concVec.tolist(),
            "gradVec": self.gradVec.tolist(),
            "config": self.config,
            "timings_": self.timings_,
            "stderr": self.stderr,
            "stdout": self.stdout,
            "ntasks": self.ntasks,
        }


def parse_ntasks(stdout: str) -> int:
    for line in stdout.splitlines():
        if m := re.match(ntasks_pattern, line):
            return int(m.group("n"))

    return 1


def parse_timings(timings: str) -> dict[str, Any]:
    f = lambda x: len(x) > 0 and "|" not in x

    header = list(map(str.strip, filter(f, timings.splitlines()[0].split("  "))))
    header[0] = "name"

    data = []
    for item_str in timings.splitlines()[2:]:
        item = list(map(str.strip, filter(f, item_str.split("  "))))
        data.append(dict(zip(header, item)))

    data.append(dict(zip(header, item)))
    return data


def load_all_data(main_path: Path):
    all_data = []
    for folder in (f for f in Path(main_path).iterdir() if f.is_dir()):
        try:
            data = load_data(folder=folder)
            print(data.mesh)
            print(f"Load data from folder {folder}")
        except FileNotFoundError as e:
            print(f"Skipping folder {folder}, due to {e}")
            continue

        all_data.append(data)
    return all_data


def plot_data(all_data: list[Data], output_folder, format: str = "png"):
    data = [d for d in all_data if d.ntasks == 1]
    fig, ax = plt.subplots(2, 4, sharex=True, sharey="row", figsize=(15, 8))

    mesh2index = {
        "1spine_mesh_coarser": 0,
        "1spine_mesh_coarser_refined_1": 1,
        "1spine_mesh_coarser_refined_2": 2,
        "1spine_mesh": 3,
    }

    dts = list(sorted({d.dt for d in data}))
    dts2color = {d: c for d, c in zip(dts, cycle(plt.cm.tab10.colors))}

    timings = [np.zeros_like(dts) for _ in mesh2index]
    lines = []
    labels = []
    for d in sorted(data, key=lambda x: x.dt):
        index = mesh2index[d.mesh]
        timings[index][dts.index(d.dt)] = d.total_run_time

        (l,) = ax[0, index].plot(d.t, d.concVec, label=d.dt, color=dts2color[d.dt])
        if index == 0:
            lines.append(l)
            labels.append(f"{d.dt:.2e}")
        ax[1, index].plot(d.t, d.gradVec, label=d.dt, color=dts2color[d.dt])
        # breakpoint()
        print(d.mesh, d.dt, d.num_refinements)#, d.folder.stem)

    for k, v in mesh2index.items():
        ax[0, v].set_title(" ".join(k.split("_")))

    ax[0, 0].set_ylabel("Average Cytosolic calcium (μM)")
    ax[1, 0].set_ylabel("Average Gradient of Cytosolic calcium (μM)")

    lgd = fig.legend(
        lines, labels, title="Time step", loc="center right", bbox_to_anchor=(1.1, 0.5)
    )
    fig.subplots_adjust(right=0.99)
    fig.savefig(
        (output_folder / "results.png").with_suffix(f".{format}"),
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )

    # Plot timings
    fig_t, ax_t = plt.subplots()

    x = np.arange(0, len(dts))
    ax_t.bar(
        x - 0.25,
        timings[0],
        width=0.25,
        label=" ".join(list(mesh2index.keys())[0].split("_")),
    )
    ax_t.bar(
        x, timings[1], width=0.25, label=" ".join(list(mesh2index.keys())[1].split("_"))
    )
    ax_t.bar(
        x + 0.25,
        timings[2],
        width=0.25,
        label=" ".join(list(mesh2index.keys())[2].split("_")),
    )
    ax_t.set_yscale("log")
    ax_t.set_xticks(x)
    ax_t.set_xticklabels(dts)
    ax_t.legend()
    ax_t.grid()
    fig_t.savefig((output_folder / "timings.png").with_suffix(f".{format}"))


def plot_refinement_study(all_data: list[Data], output_folder, format: str = "png"):
    data = sorted(
        [d for d in all_data if "coarser" in d.mesh], key=lambda x: x.num_refinements
    )
    dts = list(sorted({d.dt for d in data}))
    num_refinements = list(sorted({d.num_refinements for d in data}))
    # Find index where we have all refinements
    ref_data = [None] * len(num_refinements)
    for d in data:
        print(d.dt, d.num_refinements)#, d.folder.stem)
        # index = num_refinements.index(d.num_refinements)
        # if np.isclose(d.dt, 0.00025):
        #     ref_data[index] = [d.t, d.concVec]

    plotted = set()
    fig, ax = plt.subplots(1, len(num_refinements), sharex=True, sharey="row", figsize=(8, 3))
    # fig_err, ax_err = plt.subplots(1, len(num_refinements), sharex=True, sharey="row", figsize=(8, 2))
    lines = []
    labels = []
    for d in data:
        # index = dts.index(d.dt)
        index = num_refinements.index(d.num_refinements)
        # key = (d.num_refinements, index)
        key = (d.dt, index)
        if key in plotted:
            continue
        plotted.add(key)
        # (l,) = ax[0, index].plot(d.t, d.concVec, label=d.num_refinements)
        (l,) = ax[index].plot(d.t, d.concVec, label=d.dt)
        # interpCur = np.interp(ref_data[index][0], d.t, d.concVec)
        # ax_err[index].plot(ref_data[index][0], 100*(interpCur-ref_data[index][1])/ref_data[index][1])
        # ax[1, index].plot(d.t, d.gradVec, label=d.num_refinements)
        # ax[0, index].set_title(f"dt = {d.dt}")
        # if index == 3:
        #     lines.append(l)
        #     labels.append(d.num_refinements)
        ax[index].set_title(f"{d.num_refinements} refinements")
        ax[index].set_xlabel("Time (s)")
        if index == 2:
            lines.append(l)
            labels.append(d.dt)
    ax[0].set_ylabel("Average Cytosolic calcium (μM)")
    # ax[1, 0].set_ylabel("Average Gradient of Cytosolic calcium (μM)")
    lgd = fig.legend(lines, labels, title="Time step (s)")
    # lgd = fig.legend(
    #     lines,
    #     labels,
    #     title="Number of refinements",
    #     loc="center right",
    #     bbox_to_anchor=(1.1, 0.5),
    # )
    # fig.subplots_adjust(right=0.9)
    fig.savefig(
        (output_folder / "refinement_study.png").with_suffix(f".{format}"),
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )


def load_timings(folder: Path) -> list[dict[str, Any]]:
    timings = (folder / "timings.txt").read_text()

    # # Read total run time from the start and end timestamp from the logs
    # logs = next(folder.glob(f"{folder.name}*stderr.txt")).read_text()
    # start_time = datetime.datetime.fromisoformat(logs.splitlines()[0].split(",")[0].strip())
    # end_time = datetime.datetime.fromisoformat(logs.splitlines()[-1].split(",")[0].strip())
    # total_run_time = (end_time - start_time).total_seconds()

    f = lambda x: len(x) > 0 and "|" not in x

    header = list(map(str.strip, filter(f, timings.splitlines()[0].split("  "))))
    header[0] = "name"

    data = []
    for item_str in timings.splitlines()[2:]:
        item = list(map(str.strip, filter(f, item_str.split("  "))))
        data.append(dict(zip(header, item)))

    # item = ["Total run time", 1] + [total_run_time] * (len(header) - 2)
    data.append(dict(zip(header, item)))
    return data


def load_data(folder: Path = Path("82094")) -> Data:
    config_file = folder / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(config_file)

    t_file = folder / "tvec.txt"
    if not t_file.is_file():
        raise FileNotFoundError(t_file)

    t = np.loadtxt(t_file)
    concVec = np.load(folder / "concVec.npy")
    gradVec = np.load(folder / "gradVec.npy")

    config = json.loads(config_file.read_text())
    timings = load_timings(folder=folder)
    try:
        stdout = (folder / f"{folder.name}-dendritic_spine-stdout.txt").read_text()
        stderr = (folder / f"{folder.name}-dendritic_spine-stderr.txt").read_text()
        ntasks = parse_ntasks(stdout=stdout)
    except FileNotFoundError:
        stdout = ""
        stderr = ""
        ntasks = 1

    return Data(
        timings_=timings,
        config=config,
        t=t,
        concVec=concVec,
        gradVec=gradVec,
        stderr=stderr,
        stdout=stdout,
        ntasks=ntasks,
        folder=folder,
    )


def plot_linf_error(all_data: list[Data], output_folder, format: str = "png", subdomains=[]):
    here = Path(__file__).parent
    import dolfin
    import sys

    sys.path.append((here / ".." / "utils").as_posix())

    import smart_analysis

    try:
        import ufl_legacy as ufl
    except ImportError:
        import ufl

    # find data indices associated with coarsest and finest meshes (with min dt)
    num_refinements = []
    dt = []
    for i in range(len(all_data)):
        num_refinements.append(all_data[i].num_refinements)
        dt.append(all_data[i].dt)
    finest_logic = np.logical_and(
        np.array(dt) == min(dt), np.array(num_refinements) == max(num_refinements)
    )
    finest_idx = np.nonzero(finest_logic)[0]
    if len(finest_idx) != 1:
        raise ValueError("Could not find the finest mesh case")
    else:
        finest_idx = finest_idx[0]
    coarsest_logic = np.logical_and(
        np.array(dt) == min(dt), np.array(num_refinements) == min(num_refinements)
    )
    coarsest_idx = np.nonzero(coarsest_logic)[0]
    if len(coarsest_idx) != 1:
        raise ValueError("Could not find the coarsest mesh case")
    else:
        coarsest_idx = coarsest_idx[0]
    coarsest_data = all_data[coarsest_idx]
    finest_data = all_data[finest_idx]
    max_errs_file = output_folder / "max_errs.txt"
    l2_errs_file = output_folder / "l2_errs.txt"
    l1_errs_file = output_folder / "l1_errs.txt"
    avg_coarsest_file = output_folder / "avg_coarsest.txt"
    avg_finest_file = output_folder / "avg_finest.txt"

    if (
            not max_errs_file.is_file()
            or not l2_errs_file.is_file()
            or not l1_errs_file.is_file()
            or not avg_coarsest_file.is_file()
            or not avg_finest_file.is_file()
        ):
        # pull out mesh files from data structure
        mesh_file_coarsest = str(
            here / ".." / "scripts" / "meshes-dendritic-spine" / f"{coarsest_data.mesh}.h5"
        )
        mesh_file_finest = str(
            here / ".." / "scripts" / "meshes-dendritic-spine" / f"{finest_data.mesh}.h5"
        )
        # Load solutions
        coarsest_solutions = smart_analysis.load_solution(
            mesh_file_coarsest, coarsest_data.folder / "Ca.h5", 0
        )

        finest_solutions = smart_analysis.load_solution(
            mesh_file_finest, finest_data.folder / "Ca.h5", 0
        )

        u_coarsest = next(iter(coarsest_solutions))
        u_finest = next(iter(finest_solutions))

        V_coarsest = u_coarsest.function_space()
        V_finest = u_finest.function_space()
        u_coarsest_interp = dolfin.Function(V_finest)
        u_err = dolfin.Function(V_finest)

        if len(subdomains) > 0:
            coarsest_mf = dolfin.MeshFunction(
                    "size_t", V_coarsest.mesh(), V_coarsest.mesh().topology().dim(), 0)
            finest_mf = dolfin.MeshFunction(
                    "size_t", V_finest.mesh(), V_finest.mesh().topology().dim(), 0)
            for k, subdomain in enumerate(subdomains):
                if len(subdomain) == 6:
                    meshes = [V_coarsest.mesh(), V_finest.mesh()]
                    mfs = [coarsest_mf, finest_mf]
                    for j in range(len(meshes)):
                        mesh = meshes[j]
                        mf = mfs[j]
                        # then defines a box to specify region of integration [x0, y0, z0, x1, y1, z1]
                        for c in dolfin.cells(mesh):
                            xCur = c.midpoint().x()
                            yCur = c.midpoint().y()
                            zCur = c.midpoint().z()
                            if (
                                xCur > subdomain[0]
                                and xCur < subdomain[3]
                                and yCur > subdomain[1]
                                and yCur < subdomain[4]
                                and zCur > subdomain[2]
                                and zCur < subdomain[5]
                            ):
                                mf[c] = k+1
            dx_coarsest = dolfin.Measure("dx", V_coarsest.mesh(),subdomain_data=coarsest_mf)
            dx_finest = dolfin.Measure("dx", V_finest.mesh(),subdomain_data=finest_mf)
            vol_coarsest = [dolfin.assemble(1.0*dx_coarsest)]
            vol_finest = [dolfin.assemble(1.0*dx_finest)]
            for k in range(len(subdomains)):
                vol_coarsest.append(dolfin.assemble(1.0*dx_coarsest(k+1)))
                vol_finest.append(dolfin.assemble(1.0*dx_finest(k+1)))
        else:
            dx_coarsest = dolfin.Measure("dx", V_coarsest.mesh())
            vol_coarsest = [dolfin.assemble(1.0*dx_coarsest)]
            dx_finest = dolfin.Measure("dx", V_finest.mesh())
            vol_finest = [dolfin.assemble(1.0*dx_finest)]

        u_err_fname = output_folder / "u_err.xdmf"
        u_err_fname.unlink(missing_ok=True)
        u_err_fname.with_suffix(".h5").unlink(missing_ok=True)
        err_file = dolfin.XDMFFile(dolfin.MPI.comm_world, str(u_err_fname))
        err_file.parameters["flush_output"] = True

        avg_coarsest = np.zeros([len(coarsest_data.t),len(subdomains)+1])
        avg_finest = np.zeros([len(finest_data.t),len(subdomains)+1])
        avg_coarsest[0][0] = dolfin.assemble(u_coarsest*dx_coarsest)/vol_coarsest[0]
        avg_finest[0][0] = dolfin.assemble(u_finest*dx_finest)/vol_finest[0]
        for k in range(len(subdomains)):
            avg_coarsest[0][k+1] = dolfin.assemble(u_coarsest*dx_coarsest(k+1))/vol_coarsest[k+1]
            avg_finest[0][k+1] = dolfin.assemble(u_finest*dx_finest(k+1))/vol_finest[k+1]

        print("Computing errors")
        max_errs = [0.0]
        l2_errs = [0.0]
        l1_errs = [0.0]
        i=1
        for u_coarsest, u_finest, t in zip(
            coarsest_solutions, finest_solutions, coarsest_data.t
        ):
            avg_coarsest[i][0] = dolfin.assemble(u_coarsest*dx_coarsest)/vol_coarsest[0]
            avg_finest[i][0] = dolfin.assemble(u_finest*dx_finest)/vol_finest[0]
            for k in range(len(subdomains)):
                avg_coarsest[i][k+1] = dolfin.assemble(u_coarsest*dx_coarsest(k+1))/vol_coarsest[k+1]
                avg_finest[i][k+1] = dolfin.assemble(u_finest*dx_finest(k+1))/vol_finest[k+1]

            for j, point in enumerate(V_finest.tabulate_dof_coordinates()):
                u_coarsest_interp.vector()[j] = u_coarsest(point)
                u_err.vector()[j] = u_coarsest_interp.vector()[j] - u_finest.vector()[j]

            err_file.write(u_err, t)

            max_errs.append(max(np.abs(u_err.vector()[:])))
            l2_errs.append(
                np.sqrt(
                    dolfin.assemble((u_coarsest_interp - u_finest) ** 2 * dx_finest)
                )
            )
            l1_errs.append(
                dolfin.assemble(
                    ufl.algebra.Abs(u_coarsest_interp - u_finest) * dx_finest
                )
            )
            print(f"Processed error data {i+1} of {len(coarsest_data.t)}")
            i+=1

        # Save as text files
        np.savetxt(max_errs_file, max_errs)
        np.savetxt(l2_errs_file, l2_errs)
        np.savetxt(l1_errs_file, l1_errs)
        np.savetxt(avg_coarsest_file, avg_coarsest)
        np.savetxt(avg_finest_file, avg_finest)

    max_errs = np.loadtxt(max_errs_file)
    l2_errs = np.loadtxt(l2_errs_file)
    l1_errs = np.loadtxt(l1_errs_file)
    avg_coarsest = np.loadtxt(avg_coarsest_file)
    avg_finest = np.loadtxt(avg_finest_file)

    # Plot errors in three subplots
    fig, ax = plt.subplots(3, 1)#, figsize=(8, 12))
    ax[0].plot(max_errs)
    ax[0].set_title("Max error")
    ax[1].plot(l2_errs)
    ax[1].set_title("L2 error")
    ax[2].plot(l1_errs)
    ax[2].set_title("L1 error")
    fig.savefig((output_folder / "errors.png").with_suffix(f".{format}"))

    # Plot calcium conc avg differences
    if len(subdomains) > 0:
        cases = []
        coarse_maxes = []
        fine_maxes = []
        coarse_avgs = []
        fine_avgs = []
        for i in range(len(subdomains)):
            cases.append(f"Region{i}")
            coarse_maxes.append(max(avg_coarsest[:,i+1]))
            fine_maxes.append(max(avg_finest[:,i+1]))
            coarse_avgs.append(np.trapz(avg_coarsest[:,i+1], coarsest_data.t)/np.ptp(coarsest_data.t))
            fine_avgs.append(np.trapz(avg_finest[:,i+1], finest_data.t)/np.ptp(finest_data.t))
        cases = tuple(cases)
        maxes = {"coarse": tuple(coarse_maxes), "fine": tuple(fine_maxes)}
        avgs = {"coarse": tuple(coarse_avgs), "fine": tuple(fine_avgs)}

        x = np.arange(len(cases))  # the label locations
        width = 0.25  # the width of the bars

        fig, ax = plt.subplots(2, 1, figsize=(4,6))
        multiplier = 0
        for attribute, measurement in maxes.items():
            offset = width * multiplier
            ax[0].bar(x + offset, measurement, width, label=attribute)
            multiplier += 1
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax[0].set_ylabel('Max calcium (μM)')
        ax[0].set_xticks(x + width/2, cases)
        ax[0].legend()

        multiplier = 0
        for attribute, measurement in avgs.items():
            offset = width * multiplier
            ax[1].bar(x + offset, measurement, width, label=attribute)
            multiplier += 1
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax[1].set_ylabel('Avg calcium (μM)')
        ax[1].set_xticks(x + width/2, cases)
        fig.savefig((output_folder / "summary_ca_error.png").with_suffix(f".{format}"))


def main(
    results_folder: Path,
    output_folder: Path,
    format: str = "png",
    skip_if_processed: bool = False,
    use_tex: bool = False,
) -> int:
    plt.rcParams["text.usetex"] = use_tex

    output_folder.mkdir(exist_ok=True, parents=True)
    results_file = output_folder / "results_dendritic_spine.json"

    if skip_if_processed and results_file.is_file():
        print(f"Load results from {results_file}")
        all_results = [Data(**r) for r in json.loads(results_file.read_text())]
    else:
        print(f"Gather results from {results_folder}")
        all_results = load_all_data(results_folder)
        print(f"Save results to {results_file.absolute()}")
        results_file.write_text(
            json.dumps([r.to_json() for r in all_results], indent=4)
        )

    plot_linf_error(all_results, output_folder, format=format)
    # plot_data(all_results, output_folder, format=format)
    # plot_refinement_study(all_results, output_folder, format=format)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dendritic_spine_args.add_dendritic_spine_postprocess_arguments(parser)
    args = vars(parser.parse_args())
    raise SystemExit(main(**args))
