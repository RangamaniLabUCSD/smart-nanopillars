from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import json
from typing import NamedTuple, Any
import numpy as np
import matplotlib.pyplot as plt

import mech_parser_args
from mechanotransduction_ode import mechanotransduction_ode_calc

var_names_all = ["Cofilin_P", "Fak", "mDia", "LaminA", "FActin", "RhoA_GTP", "mDia_A", "NPC_A", "GActin", "NPC",
        "ROCK_A", "Myo", "Cofilin_NP", "LaminA_p", "YAPTAZ_nuc", "pFak", "YAPTAZ_phos", "YAPTAZ", "RhoA_GDP", "LIMK",
        "Myo_A", "ROCK", "Positionboolean", "LIMK_A", "MRTF", "MRTF_nuc"]
ode_fac_idx = var_names_all.index("FActin")
ode_yap_idx = var_names_all.index("YAPTAZ_nuc")


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
        return self.config.get("well_mixed", True)
    
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


def get_ode_solution(e_val):
    nuc_vol = 70.6
    nm_area = 58.5
    Ac = 133

    geoParam = [1925.03/4, nuc_vol, 1294.5/4, nm_area, Ac]

    return mechanotransduction_ode_calc([0, 10000], e_val, geoParam)


def plot_data(all_data: list[Data], output_folder, format: str = "png"):

    data = [d for d in all_data if np.isclose(d.e_val, 70000000.0) and d.well_mixed]
    refinements = sorted(set(d.refinement for d in data))

    # Eval vs Cutoff
    x = np.arange(len(refinements))

    fig, ax = plt.subplots(2, 1, sharex=True)
    lines = []
    labels = []
    timings = []
    for d in sorted(data, key=lambda x: x.refinement):
        l, = ax[0].plot(d.t, d.yap, label=f"refinement = {d.refinement}")
        ax[1].plot(d.t, d.fac, label=f"refinement = {d.refinement}")
        lines.append(l)
        labels.append(f"{d.refinement}")

        timings.append(float(d.total_run_time))

    # Plot ODE soltution
    t_ode, ode_results = get_ode_solution(d.e_val)
    ax[0].plot(t_ode, ode_results[:,ode_yap_idx], linestyle='dashed', label=f"ODE solution")
    ax[1].plot(t_ode, ode_results[:,ode_fac_idx], linestyle='dashed', label=f"ODE solution")
    
    ax[0].set_ylabel("YAP/TAZ")
    ax[1].set_ylabel("F-Actin")
    ax[1].set_xlabel("Time")
    lgd = fig.legend(lines, labels, title="Refinement", loc="center right", bbox_to_anchor=(1.1, 0.5))
    fig.subplots_adjust(right=0.99)
    fig.savefig((output_folder / "results.png").with_suffix(f".{format}"), bbox_extra_artists=(lgd,), bbox_inches="tight")
    
    fig_t, ax_t = plt.subplots()    
    ax_t.bar(x, timings)
    # ax_t.set_yscale("log")
    ax_t.set_xticks(x)
    ax_t.set_xticklabels(refinements)
    ax_t.set_xlabel("Refinement")
    ax_t.set_ylabel("Total run time [s]")
    ax_t.set_title("Total run time vs refinement")
    fig_t.savefig((output_folder / "timings.png").with_suffix(f".{format}"))
        
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
