from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import json
from typing import NamedTuple, Any
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

import mech_parser_args


class Data(NamedTuple):
    timings_: dict[str, Any]
    config: dict[str, Any]
    t: np.ndarray
    yap: np.ndarray
    fac: np.ndarray
    
    @property
    def refinement(self):
        return int(self.config["mesh_file"].split("refined_")[-1][0])
    
    @property
    def e_val(self):
        return self.config["e_val"]
    
    @property
    def z_cutoff(self):
        return self.config["z_cutoff"]
    
    @property
    def well_mixed(self):
        return self.config.get("well_mixed", False)
    
    @property
    def dt(self) -> float:
        return self.config["solver"]["initial_dt"]
    
    @property
    def timings(self):
        return pd.DataFrame(self.timings_)
    
    @property
    def total_run_time(self) -> float:
        return self.timings[self.timings["name"] == "mechanotransduction-example"]["wall tot"].values[0]
    
    def to_json(self) -> dict[str, Any]:
        return {
            "t": self.t.tolist(),
            "yap": self.yap.tolist(),
            "fac": self.fac.tolist(),
            "config": self.config,
            "timings_": self.timings_,
        }



def load_all_data(main_path: Path):
    all_data = []
    for folder in (f for f in Path(main_path).iterdir() if f.is_dir()):
        
        try:
            data = load_data(folder=folder)
            print(f"Load data from folder {folder}")
        except FileNotFoundError as e:
            print(f"Skipping folder {folder}, due to {e}")
            continue
        all_data.append(data)
    return all_data

def is_default(d: Data) -> bool:
    return np.isclose(d.dt, 0.01) and d.refinement == 0

def plot_data(data: list[Data], output_folder, format: str = "png"):
    fig, ax = plt.subplots(2, 3, sharex=True, sharey="row", figsize=(12, 8))
    fig_t, ax_t = plt.subplots(1, 3, sharey=True, figsize=(12, 4))    
    linestyles = [cycle(["-", "--", ":", "-."]) for _ in range(3)]

    # Plot 

    for d in data:
        print(d.refinement, d.dt, d.e_val, d.z_cutoff, d.well_mixed)

    # Get the default data
    default_data = next(d for d in data if is_default(d))

    # Plot the temporal convergence
    temporal_convergence_data = sorted([default_data] + [di for di in data if not is_default(di) and not np.isclose(di.dt, 0.01)], key=lambda x: x.dt)
    for d in temporal_convergence_data:
        ax[0, 0].plot(d.t, d.yap, linestyle=next(linestyles[0]), label=f'dt={d.dt}')
        ax[1, 0].plot(d.t, d.fac, linestyle=next(linestyles[0]), label=f'dt={d.dt}')

    x = np.arange(len(temporal_convergence_data))
    ax_t[0].bar(x, [d.total_run_time for d in temporal_convergence_data])
    ax_t[0].set_xticks(x)
    ax_t[0].set_xticklabels([d.dt for d in temporal_convergence_data])
    ax_t[0].set_xlabel("dt")
    ax_t[0].set_title("Temporal convergence")
    ax_t[0].set_ylabel("Time [s]")
        
    # Plot the spatial convergence
    spatial_convergence_data = sorted([default_data] + [di for di in data if not is_default(di) and di.refinement > 0], key=lambda x: x.refinement)
    for d in spatial_convergence_data:
        ax[0, 1].plot(d.t, d.yap, linestyle=next(linestyles[1]), label=f'# refinements = {d.refinement}')
        ax[1, 1].plot(d.t, d.yap, linestyle=next(linestyles[1]), label=f'# refinements = {d.refinement}')

    x = np.arange(len(spatial_convergence_data))
    ax_t[1].bar(x, [d.total_run_time for d in spatial_convergence_data])
    ax_t[1].set_xticks(x)
    ax_t[1].set_xticklabels([d.refinement for d in spatial_convergence_data])
    ax_t[1].set_xlabel("# refinements")
    ax_t[1].set_title("Spatial convergence")
    Path(output_folder).mkdir(exist_ok=True, parents=True)
    fig.savefig((Path(output_folder) / "mechanotransduction_yap.png").with_suffix(f".{format}"))
    fig_t.savefig((Path(output_folder) / "timings_mechanotransduction.png").with_suffix(f".{format}"))


        
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
    yap = np.loadtxt(folder / "YAPTAZ_nuc.txt")
    fac = np.loadtxt(folder / "FActin.txt")

    config = json.loads(config_file.read_text())
    timings = load_timings(folder=folder)

    return Data(timings_=timings, config=config, t=t, yap=yap, fac=fac)
    
def main(results_folder: Path, output_folder: Path, 
         format: str = "png",
         skip_if_processed: bool = False,
         use_tex: bool = False,
    ) -> int:
    
    plt.rcParams["text.usetex"] = use_tex

    output_folder.mkdir(exist_ok=True, parents=True)
    results_file = output_folder / "results_mechanotransduction.json"

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
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mech_parser_args.add_mechanotransduction_postprocess_arguments(parser)
    args = vars(parser.parse_args())
    raise SystemExit(main(**args))
