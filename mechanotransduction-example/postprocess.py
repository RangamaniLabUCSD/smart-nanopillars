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
from mechanotransduction_ode import mechanotransduction_ode_calc


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


def plot_data(data: list[Data], output_folder, format: str = "png"):
    fig, ax = plt.subplots(2, 3, sharex=True, sharey="row", figsize=(12, 8))
    fig_t, ax_t = plt.subplots(1, 3, sharey=True, figsize=(12, 4))    
    linestyles = [cycle(["-", "--", ":", "-."]) for _ in range(3)]

    # Plot 
    refinements = set()
    e_vals = set()
    z_cutoffs = set()

    for d in data:
        print(d.refinement, d.dt, d.e_val, d.z_cutoff, d.well_mixed)
        refinements.add(d.refinement)
        e_vals.add(d.e_val)
        z_cutoffs.add(d.z_cutoff)

    e_vals = sorted(list(e_vals))
    z_cutoffs = sorted(list(z_cutoffs))
    refinements = sorted(list(refinements))
   
    # Eval vs Cutoff
    x = np.arange(len(refinements))
    fig_yap, ax_yap = plt.subplots(len(e_vals), 2, sharex=True, sharey="row", figsize=(10, 10))
    fig_fac, ax_fac = plt.subplots(len(e_vals), 2, sharex=True, sharey="row", figsize=(10, 10))
    fig_t, ax_t = plt.subplots(len(e_vals), 2, sharex=True, sharey="row", figsize=(10, 10))
    for i, e_val in enumerate(e_vals):
        ax2 = ax_yap[i, -1].twinx()
        ax2.set_yticks([])
        ax2.set_ylabel(f"e val = {e_val}")

        ax2 = ax_fac[i, -1].twinx()
        ax2.set_yticks([])
        ax2.set_ylabel(f"e val = {e_val}")

        ax2 = ax_t[i, -1].twinx()
        ax2.set_yticks([])
        ax2.set_ylabel(f"e val = {e_val}")
        for j, well_mixed in enumerate([True, False]):
            ax_yap[0, j].set_title(f"well mixed = {well_mixed}")
            ax_fac[0, j].set_title(f"well mixed = {well_mixed}")
            ax_t[0, j].set_title(f"well mixed = {well_mixed}")
            times = []
       
            for d in sorted([di for di in data if di.e_val == e_val and di.well_mixed is well_mixed], key=lambda x: x.refinement):
                times.append(d.total_run_time)
                ax_yap[i, j].plot(d.t, d.yap, label=f"refinement = {d.refinement}")
                ax_fac[i, j].plot(d.t, d.fac, label=f"refinement = {d.refinement}")
            
            if well_mixed:
                nuc_vol = 70.6
                nm_area = 58.5
                Ac = 133
                # geoParam = [cyto_vol, nuc_vol, pm_area, nm_area, Ac]
                geoParam = [1925.03/4, nuc_vol, 1294.5/4, nm_area, Ac]
                var_names_all = ["Cofilin_P", "Fak", "mDia", "LaminA", "FActin", "RhoA_GTP", "mDia_A", "NPC_A", "GActin", "NPC",
                        "ROCK_A", "Myo", "Cofilin_NP", "LaminA_p", "YAPTAZ_nuc", "pFak", "YAPTAZ_phos", "YAPTAZ", "RhoA_GDP", "LIMK",
                        "Myo_A", "ROCK", "Positionboolean", "LIMK_A", "MRTF", "MRTF_nuc"]
                ode_fac_idx = var_names_all.index("FActin")
                ode_yap_idx = var_names_all.index("YAPTAZ_nuc")
                # ode_yap_idx = [var_names_all.index(name) for name in ["YAPTAZ_nuc", "YAPTAZ_phos", "YAPTAZ"]]
                # plot ode solution
                t_ode, ode_results = mechanotransduction_ode_calc([0, 10000], e_val, geoParam)
                ax_yap[i, j].plot(t_ode, ode_results[:,ode_yap_idx],#/(ode_results[:,ode_yap_idx[1]]+ode_results[:,ode_yap_idx[2]]), 
                                linestyle='dashed', label=f"ODE solution")
                ax_fac[i, j].plot(t_ode, ode_results[:,ode_fac_idx], 
                                linestyle='dashed', label=f"ODE solution")

            ax_t[i, j].bar(x, times, width=0.3)
            ax_t[i, j].set_yscale("log")
            ax_t[i, j].set_xticks(x)
            ax_t[i, j].set_xticklabels(list(sorted(refinements)))
            if i == len(e_vals) - 1:
                ax_t[i, j].set_xlabel("Refinement")

            ax_yap[i, j].legend()
            ax_fac[i, j].legend()
            ax_t[i, j].legend()

    fig_yap.savefig((Path(output_folder) / "yap").with_suffix(f".{format}"))
    fig_fac.savefig((Path(output_folder) / "fac").with_suffix(f".{format}"))
    fig_t.savefig((Path(output_folder) / "timings").with_suffix(f".{format}"))
    

        
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
