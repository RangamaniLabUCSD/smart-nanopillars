from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import json
from typing import NamedTuple, Any
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

import dendritic_spine_args


class Data(NamedTuple):
    timings_: dict[str, Any]
    config: dict[str, Any]
    t: np.ndarray
    concVec: np.ndarray
    gradVec: np.ndarray
    
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
            return self.timings[self.timings["name"] == "dendritic-spine-example"]["wall tot"].values[0]
        except IndexError:
            return np.nan
    
    def to_json(self) -> dict[str, Any]:
        return {
            "t": self.t.tolist(),
            "concVec": self.concVec.tolist(),
            "gradVec": self.gradVec.tolist(),
            "config": self.config,
            "timings_": self.timings_,
        }



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

    data = [d for d in all_data if d.num_refinements == 0]
    fig, ax = plt.subplots(2, 3, sharex=True, sharey="row", figsize=(12, 8))
      
    mesh2index = {"1spine_mesh_coarser": 0, "1spine_mesh": 1, "2spine_mesh": 2}

    dts = list(sorted({d.dt for d in data}))
    timings = (np.zeros_like(dts), np.zeros_like(dts), np.zeros_like(dts))
    # Plot
    for d in sorted(data, key=lambda x: x.dt):
        index = mesh2index[d.mesh]
        timings[index][dts.index(d.dt)] = d.total_run_time
        ax[0, index].plot(d.t, d.concVec, label=d.dt)
        ax[1, index].plot(d.t, d.gradVec, label=d.dt)
        # breakpoint()
        print(d.mesh, d.dt, d.total_run_time)


    for k, v in mesh2index.items():
        ax[0, v].set_title(" ".join(k.split("_")))
    
    ax[0, 0].set_ylabel("Average Cytosolic calcium (μM)")
    ax[1, 0].set_ylabel("Average Gradient of Cytosolic calcium (μM)")

    for axi in ax.flatten():
        axi.legend(title="Timestep")

    fig.savefig((output_folder / "results.png").with_suffix(f".{format}"))

    # Plot timings
    fig_t, ax_t = plt.subplots()

    x = np.arange(0, len(dts))
    ax_t.bar(x-0.25, timings[0], width=0.25, label=" ".join(list(mesh2index.keys())[0].split("_")))
    ax_t.bar(x, timings[1], width=0.25, label=" ".join(list(mesh2index.keys())[1].split("_")))
    ax_t.bar(x+0.25, timings[2], width=0.25, label=" ".join(list(mesh2index.keys())[2].split("_")))
    ax_t.set_yscale("log")
    ax_t.set_xticks(x)
    ax_t.set_xticklabels(dts)
    ax_t.legend()
    ax_t.grid()
    fig_t.savefig((output_folder / "timings.png").with_suffix(f".{format}"))


def plot_refinement_study(all_data: list[Data], output_folder, format: str = "png"):
    data = sorted([d for d in all_data if "coarser" in d.mesh], key=lambda x: x.num_refinements)
    dts = list(sorted({d.dt for d in data}))
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
        l, = ax[0, index].plot(d.t, d.concVec, label=d.num_refinements)
        ax[1, index].plot(d.t, d.gradVec, label=d.num_refinements)
        ax[0, index].set_title(f"dt = {d.dt}")
        if index == 0:
            lines.append(l)
            labels.append(d.num_refinements)
    ax[0, 0].set_ylabel("Average Cytosolic calcium (μM)")
    ax[1, 0].set_ylabel("Average Gradient of Cytosolic calcium (μM)")
    lgd = fig.legend(lines, labels, title="Number of refinements", loc="center right", bbox_to_anchor=(1.1, 0.5))
    fig.subplots_adjust(right=0.9)
    fig.savefig((output_folder / "refinement_study.png").with_suffix(f".{format}"), bbox_extra_artists=(lgd,), bbox_inches="tight")
        
def load_timings(folder: Path) -> dict[str, Any]:
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

    return Data(timings_=timings, config=config, t=t, concVec=concVec, gradVec=gradVec)
    
def main(results_folder: Path, output_folder: Path, 
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
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dendritic_spine_args.add_dendritic_spine_postprocess_arguments(parser)
    args = vars(parser.parse_args())
    raise SystemExit(main(**args))
