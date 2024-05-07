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
    petsc_timings_: list[dict[str, Any]]
    config: dict[str, Any]
    t: np.ndarray
    concVec: np.ndarray
    gradVec: np.ndarray
    stderr: str
    stdout: str
    ntasks: int
    folder: Path

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
    def petsc_timings(self):
        return pd.DataFrame(self.petsc_timings_)

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
            "petsc_timings": self.petsc_timings_,
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
        print(d.mesh, d.dt, d.num_refinements, d.folder.stem)

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
    for d in data:
        print(d.dt, d.num_refinements, d.folder.stem)

    plotted = set()
    fig, ax = plt.subplots(2, len(dts), sharex=True, sharey="row", figsize=(12, 8))
    lines = []
    labels = []
    for d in data:
        index = dts.index(d.dt)
        key = (d.num_refinements, index)
        if key in plotted:
            continue
        plotted.add(key)
        (l,) = ax[0, index].plot(d.t, d.concVec, label=d.num_refinements)
        ax[1, index].plot(d.t, d.gradVec, label=d.num_refinements)
        ax[0, index].set_title(f"dt = {d.dt}")
        if index == 3:
            lines.append(l)
            labels.append(d.num_refinements)
    ax[0, 0].set_ylabel("Average Cytosolic calcium (μM)")
    ax[1, 0].set_ylabel("Average Gradient of Cytosolic calcium (μM)")
    lgd = fig.legend(
        lines,
        labels,
        title="Number of refinements",
        loc="center right",
        bbox_to_anchor=(1.1, 0.5),
    )
    fig.subplots_adjust(right=0.9)
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

    if (petsc_timings_file := folder / "petsc_timings.py").is_file():
        petsc_timings_ = {}
        exec(petsc_timings_file.read_text(), petsc_timings_)
        stages = petsc_timings_.pop("Stages")["Main Stage"]
        petsc_timings = []
        for k, v in stages.items():
            petsc_timings.append({"name": k, **v[0]})

    else:
        petsc_timings = []

    config = json.loads(config_file.read_text())
    timings = load_timings(folder=folder)
    if (stdout_file := folder / f"{folder.name}-dendritic_spine-stdout.txt").is_file():
        stdout = stdout_file.read_text()
        ntasks = parse_ntasks(stdout=stdout)
    else:
        stdout = ""
        ntasks = 1

    if (stderr_file := folder / f"{folder.name}-dendritic_spine-stderr.txt").is_file():
        stderr = stderr_file.read_text()
    else:
        stderr = ""

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
        petsc_timings_=petsc_timings,
    )


def plot_timings(all_data: list[Data], output_folder, format: str = "png"):
    data = sorted(
        [
            d
            for d in all_data
            if "coarser" in d.mesh and np.isclose(d.dt, 0.001) and d.ntasks == 1
        ],
        key=lambda x: x.num_refinements,
    )

    breakpoint()

    keys = [
        "SNES Assemble Jacobian Nested Matrix",
        "SNES Assemble Residual Nest Vector",
        "Initialize Model",
        # "dendritic-spine-example"
    ]

    petsc_keys = [
        "SNESSolve",
        "KSPSolve",
        "SnesJacobianEval",
        "SnesFunctionEval",
        "PCApply",
    ]

    names = data[0].timings["name"].values.tolist()

    # Create bar plot
    fig, ax = plt.subplots()
    indices = []
    for name in keys:
        indices.append(names.index(name))
    indices = np.array(indices)
    x = np.arange(0, 3)
    width = 0.25
    bottom = np.zeros_like(x)
    # breakpoint()
    for index, key in zip(indices, keys):
        y = np.zeros_like(x)
        for i, d in enumerate(data):
            y[i] = float(d.timings["wall tot"].values[index])

        print(key, y, bottom)
        ax.bar(x - width, y, width=width, label=key, bottom=bottom)
        bottom += y

    index = names.index("[MixedAssembler] Assemble cells")
    y = [float(d.timings["wall tot"].values[index]) for d in data]
    ax.bar(x, y, width=width, label="Assemble cells")

    index = names.index("dendritic-spine-example")
    y = [float(d.timings["wall tot"].values[index]) for d in data]
    ax.bar(x + width, y, width=width, label="Total run time")

    ax.set_xticks(x)
    ax.legend()
    fig.savefig((output_folder / "timings_profile.png").with_suffix(f".{format}"))


def plot_petsc_timings(all_data: list[Data], output_folder, format: str = "png"):
    data = sorted(
        [
            d
            for d in all_data
            if "coarser" in d.mesh and np.isclose(d.dt, 0.001) and d.ntasks == 1
        ],
        key=lambda x: x.num_refinements,
    )

    keys = [
        "KSPSolve",
        "SNESJacobianEval",
        "SNESFunctionEval",
        "PCApply",
    ]

    names = data[0].petsc_timings["name"].values.tolist()

    # Create bar plot
    fig, ax = plt.subplots()
    indices = []
    for name in keys:
        indices.append(names.index(name))
    indices = np.array(indices)
    x = np.arange(0, 3)
    width = 0.25
    bottom = np.zeros_like(x)
    # breakpoint()
    for index, key in zip(indices, keys):
        y = np.zeros_like(x)
        for i, d in enumerate(data):
            y[i] = float(d.petsc_timings["time"].values[index])

        print(key, y, bottom)
        ax.bar(x - width / 2, y, width=width, label=key, bottom=bottom)
        bottom += y

    index = names.index("SNESSolve")
    y = [float(d.petsc_timings["time"].values[index]) for d in data]
    ax.bar(x + width / 2, y, width=width, label="SNESSolve")

    # index = names.index("dendritic-spine-example")
    # y = [float(d.petsc_timings["time"].values[index]) for d in data]
    # ax.bar(x + width, y, width=width, label="Total run time")

    ax.set_xticks(x)
    ax.legend()
    fig.savefig((output_folder / "petsc_timings_profile.png").with_suffix(f".{format}"))


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

    # plot_data(all_results, output_folder, format=format)
    # plot_refinement_study(all_results, output_folder, format=format)
    # plot_timings(all_results, output_folder, format=format)
    plot_petsc_timings(all_results, output_folder, format=format)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dendritic_spine_args.add_dendritic_spine_postprocess_arguments(parser)
    args = vars(parser.parse_args())
    raise SystemExit(main(**args))
