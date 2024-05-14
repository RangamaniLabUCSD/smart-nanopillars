from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import json
from dataclasses import dataclass
from typing import Any
from itertools import cycle
import numpy as np
import re
import matplotlib.pyplot as plt

import dendritic_spine_args


ntasks_pattern = re.compile("ntasks: (?P<n>\d+)")


@dataclass
class Data:
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
        if not hasattr(self, "_timings"):
            self._timings = pd.DataFrame(self.timings_)
            # Now update all dtypes
            self._timings["reps"] = self._timings["reps"].astype(int)
            self._timings["wall avg"] = self._timings["wall avg"].astype(float)
            self._timings["wall tot"] = self._timings["wall tot"].astype(float)
            self._timings["usr avg"] = self._timings["usr avg"].astype(float)
            self._timings["usr tot"] = self._timings["usr tot"].astype(float)
            self._timings["sys avg"] = self._timings["sys avg"].astype(float)
            self._timings["sys tot"] = self._timings["sys tot"].astype(float)
            self._timings["name"] = self._timings["name"].astype(str)

        return self._timings

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

    key_list = [
        (
            "petsc_timings",
            [
                "KSPSolve",
                "SNESLineSearch",
                "SNESJacobianEval",
                "SNESFunctionEval",
            ],
        ),
        (
            "timings",
            [
                "Solve loop [main]",
                "Setup [main]",
                "Initialize Model [main]",
            ],
        ),
    ]

    data_refined0 = [
        d for d in all_data if "coarser" in d.mesh and d.num_refinements == 0
    ]
    data_refined1 = [
        d for d in all_data if "coarser" in d.mesh and d.num_refinements == 1
    ]
    data_refined2 = [
        d for d in all_data if "coarser" in d.mesh and d.num_refinements == 2
    ]

    timings_refined0 = [d.timings for d in data_refined0]
    timings_refined0_concat = pd.concat(tuple(timings_refined0))
    timings_refined0_mean = timings_refined0_concat.groupby("name").mean()
    timings_refined0_std = timings_refined0_concat.groupby("name").std()

    petsc_timings_refined0 = [d.petsc_timings for d in data_refined0]
    petsc_timings_refined0_concat = pd.concat(tuple(petsc_timings_refined0))
    petsc_timings_refined0_mean = petsc_timings_refined0_concat.groupby("name").mean()
    petsc_timings_refined0_std = petsc_timings_refined0_concat.groupby("name").std()

    timings_refined1 = [d.timings for d in data_refined1]
    timings_refined1_concat = pd.concat(tuple(timings_refined1))
    timings_refined1_mean = timings_refined1_concat.groupby("name").mean()
    timings_refined1_std = timings_refined1_concat.groupby("name").std()

    petsc_timings_refined1 = [d.petsc_timings for d in data_refined1]
    petsc_timings_refined1_concat = pd.concat(tuple(petsc_timings_refined1))
    petsc_timings_refined1_mean = petsc_timings_refined1_concat.groupby("name").mean()
    petsc_timings_refined1_std = petsc_timings_refined1_concat.groupby("name").std()

    timings_refined2 = [d.timings for d in data_refined2]
    timings_refined2_concat = pd.concat(tuple(timings_refined2))
    timings_refined2_mean = timings_refined2_concat.groupby("name").mean()
    timings_refined2_std = timings_refined2_concat.groupby("name").std()

    petsc_timings_refined2 = [d.petsc_timings for d in data_refined2]
    petsc_timings_refined2_concat = pd.concat(tuple(petsc_timings_refined2))
    petsc_timings_refined2_mean = petsc_timings_refined2_concat.groupby("name").mean()
    petsc_timings_refined2_std = petsc_timings_refined2_concat.groupby("name").std()

    petsc_names = list(petsc_timings_refined0_mean.index)
    names = list(timings_refined0_mean.index)

    replace_dict = {
        "SNES Assemble Jacobian Nested Matrix": "Assemble Jacobian",
        "SNES Assemble Residual Nest Vector": "Assemble Residual",
    }

    # Create bar plot
    fig, ax = plt.subplots()
    indices_list = []
    for df_name, keys in key_list:
        inds = []
        for name in keys:
            if df_name == "timings":
                inds.append(names.index(name))
            else:
                inds.append(petsc_names.index(name))
        indices_list.append(inds)
    x = np.arange(0, 3)
    width = 0.4
    bottom = np.zeros_like(x)

    colors = plt.cm.tab10.colors

    lines = []
    labels = []
    i = 0
    y_line_search = np.zeros_like(x)
    err_line_search = np.zeros_like(x)
    for j, (indices, (df_name, keys)) in enumerate(zip(indices_list, key_list)):
        for index, key in zip(indices, keys):
            y = np.zeros_like(x)
            err = np.zeros_like(x)
            if df_name == "timings":
                for k, (timings_mean, timings_std) in enumerate(
                    [
                        (timings_refined0_mean, timings_refined0_std),
                        (timings_refined1_mean, timings_refined1_std),
                        (timings_refined2_mean, timings_refined2_std),
                    ]
                ):
                    y[k] = float(timings_mean["wall tot"].values[index])
                    err[k] = float(timings_std["wall tot"].values[index])
            else:
                for k, (timings_mean, timings_std) in enumerate(
                    [
                        (petsc_timings_refined0_mean, petsc_timings_refined0_std),
                        (petsc_timings_refined1_mean, petsc_timings_refined1_std),
                        (petsc_timings_refined2_mean, petsc_timings_refined2_std),
                    ]
                ):
                    y[k] = float(timings_mean["time"].values[index])
                    err[k] = float(timings_std["time"].values[index])

                if key == "SNESLineSearch":
                    y_line_search[:] = y.copy()
                    err_line_search[:] = err.copy()

                if key == "SNESFunctionEval":
                    # Line search is part of the function eval
                    y -= y_line_search
                    err = np.sqrt(err**2 - err_line_search**2)

            print(key, y, bottom)
            l = ax.bar(
                x + (2 * j - 1) * width / 2,
                y,
                width=width,
                bottom=bottom,
                yerr=err,
                capsize=5,
                color=colors[i],
            )
            lines.append(l)
            label = replace_dict.get(key, key).strip(" [main]")
            labels.append(label)
            bottom += y
            i += 1
        bottom[:] = 0

    ax.set_xticks(x)

    # Put legend outside of the plot to the right
    lgd = fig.legend(
        lines,
        labels,
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        title="Timings",
        title_fontsize="large",
    )
    ax.set_yscale("log")

    ax.set_xlabel("Number of refinements")
    fig.savefig(
        (output_folder / "timings_profile.png").with_suffix(f".{format}"),
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )
    fig.savefig(
        (output_folder / "timings_profile.svg").with_suffix(f".{format}"),
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )


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

    plot_data(all_results, output_folder, format=format)
    plot_refinement_study(all_results, output_folder, format=format)
    plot_timings(all_results, output_folder, format=format)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dendritic_spine_args.add_dendritic_spine_postprocess_arguments(parser)
    args = vars(parser.parse_args())
    raise SystemExit(main(**args))
