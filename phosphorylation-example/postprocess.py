# ## Compare model results to analytical solution and previous results
#
# Here, we plot the steady-state concentration as a function of cell radius, according to the analytical solution and the SMART numerical solution. The analytical solution for the average concentration in the cytosol at steady state is given in Meyers and Odde 2006 and is included here for ease of reference:
#
# $$
# \bigl< c_{A_{phos}} \bigr> = \frac{6C_1}{R} \left[ \frac{\cosh{\Phi}}{\Phi} - \frac{\sinh{\Phi}}{\Phi^2} \right]\\
# \text{where} \quad C_1 = \frac{k_{kin} c_{Tot} R^2}{\left[3D(1/L_{gradient} - 1/R) + k_{kin}R \right] e^\Phi + \left[3D(1/L_{gradient} + 1/R) - k_{kin}R \right] e^{-\Phi}}\\
# \text{and} \quad \Phi = \frac{R}{L_{gradient}} \quad \text{and} \quad L_{gradient} = \sqrt{\frac{D_{A_{phos}}}{k_p}}
# $$
from typing import NamedTuple, Dict, Any
from pathlib import Path
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re
from itertools import cycle
import json
import phosphorylation_parser_args

ntasks_pattern = re.compile("ntasks: (?P<n>\d)")
cmap = plt.get_cmap("tab10")
linestyles = ['-','--','-.',':']
markers = ["", ".", "o"]

hmin_hmax = {
    "curRadius_10_axisymmetric_refined_0": {
        "hmax": 2.107398697464825,
        "hmin": 0.18146878678974115,
    },
    "curRadius_10_axisymmetric_refined_1": {
        "hmax": 1.203146011049625,
        "hmin": 0.09073439339487056,
    },
    "curRadius_10_axisymmetric_refined_2": {
        "hmax": 0.6015730055248131,
        "hmin": 0.04536719669743526,
    },
    "curRadius_10_refined_0": {"hmax": 2.9569505547636203, "hmin": 0.18243415650338052},
    "curRadius_10_refined_1": {"hmax": 2.0282963176600526, "hmin": 0.08695003394032726},
    "curRadius_10_refined_2": {"hmax": 1.1324701545883051, "hmin": 0.04347501697016299},
    "curRadius_1_axisymmetric_refined_0": {
        "hmax": 0.26490778698922435,
        "hmin": 0.16056605553927863,
    },
    "curRadius_1_axisymmetric_refined_1": {
        "hmax": 0.1581009706925238,
        "hmin": 0.0802830277696392,
    },
    "curRadius_1_axisymmetric_refined_2": {
        "hmax": 0.0790504853462619,
        "hmin": 0.0401415138848195,
    },
    "curRadius_1_refined_0": {"hmax": 0.4100302314099659, "hmin": 0.21629758636837643},
    "curRadius_1_refined_1": {"hmax": 0.3022104956209887, "hmin": 0.10814879318418821},
    "curRadius_1_refined_2": {"hmax": 0.1713261905906801, "hmin": 0.05278417512311456},
    "curRadius_2_axisymmetric_refined_0": {
        "hmax": 0.4180690194683049,
        "hmin": 0.1677446599326136,
    },
    "curRadius_2_axisymmetric_refined_1": {
        "hmax": 0.27392365842436256,
        "hmin": 0.08387232996630663,
    },
    "curRadius_2_axisymmetric_refined_2": {
        "hmax": 0.1369618292121815,
        "hmin": 0.04193616498315315,
    },
    "curRadius_2_refined_0": {"hmax": 0.6745129390201708, "hmin": 0.2064875125954894},
    "curRadius_2_refined_1": {"hmax": 0.4406744161036693, "hmin": 0.10324375629774465},
    "curRadius_2_refined_2": {
        "hmax": 0.24875502991800244,
        "hmin": 0.051621878148872244,
    },
    "curRadius_4_axisymmetric_refined_0": {
        "hmax": 0.826381779846353,
        "hmin": 0.17537383439392057,
    },
    "curRadius_4_axisymmetric_refined_1": {
        "hmax": 0.5127250117828788,
        "hmin": 0.08768691719696028,
    },
    "curRadius_4_axisymmetric_refined_2": {
        "hmax": 0.2563625058914398,
        "hmin": 0.043843458598478394,
    },
    "curRadius_4_refined_0": {"hmax": 1.1675982278230856, "hmin": 0.19354524111099},
    "curRadius_4_refined_1": {"hmax": 0.795009636638825, "hmin": 0.096772620555495},
    "curRadius_4_refined_2": {"hmax": 0.4838682319356164, "hmin": 0.048386310277747334},
    "curRadius_6_axisymmetric_refined_0": {
        "hmax": 1.1186753394210012,
        "hmin": 0.17578088012385903,
    },
    "curRadius_6_axisymmetric_refined_1": {
        "hmax": 0.5593376697105006,
        "hmin": 0.0878904400619295,
    },
    "curRadius_6_axisymmetric_refined_2": {
        "hmax": 0.27966883485525074,
        "hmin": 0.04394522003096466,
    },
    "curRadius_6_refined_0": {"hmax": 1.7475830500863683, "hmin": 0.18994236725031408},
    "curRadius_6_refined_1": {"hmax": 1.3511624344614612, "hmin": 0.09497118362515661},
    "curRadius_6_refined_2": {"hmax": 0.751970864469718, "hmin": 0.04748559181257788},
    "curRadius_8_axisymmetric_refined_0": {
        "hmax": 1.5548494560250927,
        "hmin": 0.18423090220778784,
    },
    "curRadius_8_axisymmetric_refined_1": {
        "hmax": 1.0521403888691427,
        "hmin": 0.09211545110389385,
    },
    "curRadius_8_axisymmetric_refined_2": {
        "hmax": 0.5260701944345721,
        "hmin": 0.04605772555194692,
    },
    "curRadius_8_refined_0": {"hmax": 2.3990626175740966, "hmin": 0.17628233360891069},
    "curRadius_8_refined_1": {"hmax": 1.7128579342869876, "hmin": 0.08814116680445533},
    "curRadius_8_refined_2": {"hmax": 0.9724037385516981, "hmin": 0.044070583402227574},
}


class Data(NamedTuple):
    t: np.ndarray
    ss: np.ndarray
    l2: np.ndarray
    config: Dict[str, Any]
    timings_: Dict[str, Any]
    stderr: str
    stdout: str
    ntasks: int

    @property
    def radius(self):
        return self.config["curRadius"]

    @property
    def dt(self):
        return self.config["time_step"]

    @property
    def diffusion(self):
        return self.config.get("diffusion", 10.0)

    @property
    def axisymmetric(self):
        return "axisymmetric" in self.config["mesh_file"]

    @property
    def timings(self):
        return pd.DataFrame(self.timings_)
    
    @property
    def hmin(self) -> float:
        _hmin = self.config.get("hmin")
        if _hmin is None:
            return  hmin_hmax[self.config["mesh_file"].split("/")[-1]]["hmin"]
        return _hmin
    @property
    def hmax(self) -> float:
        _hmax = self.config.get("hmax")
        if _hmax is None:
            return hmin_hmax[self.config["mesh_file"].split("/")[-1]]["hmax"]
        return _hmax

    @property
    def refinement(self):
        return int(self.config["mesh_file"].split("refined_")[-1][0])

    def __str__(self):
        return f"radius={self.radius}, dt={self.dt}, refinement={self.refinement}"

    def to_json(self) -> Dict[str, Any]:
        return {
            "t": self.t.tolist(),
            "ss": self.ss.tolist(),
            "l2": float(self.l2),
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


def parse_timings(timings: str) -> Dict[str, Any]:
    f = lambda x: len(x) > 0 and "|" not in x

    header = list(map(str.strip, filter(f, timings.splitlines()[0].split("  "))))
    header[0] = "name"

    data = []
    for item_str in timings.splitlines()[2:]:
        item = list(map(str.strip, filter(f, item_str.split("  "))))
        data.append(dict(zip(header, item)))

    data.append(dict(zip(header, item)))
    return data


def load_results(folder):
    data = []
    for result_folder in Path(folder).iterdir():
        if not result_folder.is_dir():
            continue
        if not (result_folder / "avg_Aphos.txt").exists():
            continue
        stdout = (
            result_folder / f"{result_folder.name}-phosphorylation-stdout.txt"
        ).read_text()
        stderr = (
            result_folder / f"{result_folder.name}-phosphorylation-stderr.txt"
        ).read_text()
        ntasks = parse_ntasks(stdout=stdout)
        config = json.loads((result_folder / "config.json").read_text())
        t = np.loadtxt(result_folder / "tvec.txt")
        ss = np.loadtxt(result_folder / "avg_Aphos.txt")
        l2 = np.loadtxt(result_folder / "L2norm.txt")
        timings = parse_timings((result_folder / "timings.txt").read_text())
        data.append(
            Data(
                t=t,
                ss=ss,
                l2=l2,
                config=config,
                timings_=timings,
                stderr=stderr,
                stdout=stdout,
                ntasks=ntasks,
            )
        )
        print(data[-1])

    if len(data) == 0:
        raise RuntimeError(f"No results found in folder {folder}")
    return data


def analytical_solution(radius, D=10.0):
    k_kin = 50
    k_p = 10
    cT = 1
    thieleMod = radius / np.sqrt(D / k_p)
    C1 = (
        k_kin
        * cT
        * radius**2
        / (
            (3 * D * (np.sqrt(k_p / D) - (1 / radius)) + k_kin * radius)
            * np.exp(thieleMod)
            + (3 * D * (np.sqrt(k_p / D) + (1 / radius)) - k_kin * radius)
            * np.exp(-thieleMod)
        )
    )

    return (6 * C1 / radius) * (
        np.cosh(thieleMod) / thieleMod - np.sinh(thieleMod) / thieleMod**2
    )


def plot_error_analytical_solution_different_radius(all_results, output_folder, format):
    
    radii = sorted({r.radius for r in all_results if r.ntasks == 1})
    diffusions = sorted({r.diffusion for r in all_results})
    refinements = sorted({r.refinement for r in all_results})
    dts =  sorted({r.dt for r in all_results})
    axisymmetrics = {r.axisymmetric for r in all_results}

    fig, ax = plt.subplots(len(radii), len(diffusions), sharex=True, figsize=(10, 10))

    for i, radius in enumerate(radii):
        ax2 = ax[i, -1].twinx()
        ax2.set_ylabel(f"Radius = {radius}")
        ax2.set_yticks([])
        
        for j, diffusion in enumerate(diffusions):
            ax[0, j].set_title(f"D = {diffusion}")

        for diffusion in diffusions:
            for axisymmetric in axisymmetrics:
                # fig_steady, ax_steady = plt.subplots()
                # fig_percent, ax_percent = plt.subplots()
                # fig_l2, ax_l2 = plt.subplots()
                # fig_time, ax_time = plt.subplots()
                for refinement in refinements:
        
                    results = list(
                        sorted(
                            filter(
                                lambda d: d.refinement == refinement
                                and np.isclose(d.radius, radius)
                                and np.isclose(d.diffusion, diffusion)
                                and (d.axisymmetric is axisymmetric),
                                all_results,
                            ),
                            key=lambda d: d.dt
                        )
                    )
                    radiusVec = np.array([d.radius for d in results])
                    cA = analytical_solution(radiusVec, D=diffusion)
                    ss_vec = np.array([d.ss[-1] for d in results])
                    percentError = 100 * np.abs(ss_vec - cA) / cA
                    l2 = np.array([d.l2 for d in results])
                    total_run_time = [
                        float(
                            result.timings[
                                result.timings["name"] == "phosphorylation-example"
                            ].iloc[0]["wall tot"]
                        )
                        for result in results
                    ]

            #     ax_steady.plot(
            #         radiusVec,
            #         ss_vec,
            #         linestyle=":",
            #         marker="o",
            #         label=f"SMART simulation (refinement: {refinement})",
            #     )
            #     ax_percent.plot(
            #         radiusVec, percentError, label=f"refinement {refinement}"
            #     )
            #     ax_l2.semilogy(radiusVec, l2, label=f"refinement {refinement}")
            #     rmse = np.sqrt(np.mean(percentError**2))
            #     print(
            #         f"RMSE with respect to analytical solution = {rmse:.3f}% for refinement {refinement}, D: {diffusion}, axisymetric: {axisymmetric}"
            #     )
            #     ax_time.plot(radiusVec, total_run_time, marker="o")

            # radiusTest = np.logspace(0, 1, 100)
            # cA_smooth = analytical_solution(radiusTest, D=diffusion)
            # ax_steady.plot(radiusTest, cA_smooth, label="Analytical solution")

            # ax_l2.set_xlabel("Cell radius (μm)")
            # ax_l2.set_ylabel("$ \| u_e - u \|^2$")
            # ax_l2.set_title("$\ell^2$ error from analytical solution")
            # ax_l2.legend()
            # fig_l2.savefig(
            #     (
            #         output_folder
            #         / f"error_radius_l2_diffusion_{diffusion}_axisymmetric_{axisymmetric}.{format}"
            #     )
            # )
            # plt.close(fig_l2)

            # ax_time.set_xlabel("Cell radius (μm)")
            # ax_time.set_ylabel("Total run time [s]")
            # ax_time.legend()
            # fig_time.savefig(
            #     output_folder
            #     / f"total_time_radius_diffusion_{diffusion}_axisymmetric_{axisymmetric}.{format}"
            # )
            # plt.close(fig_time)

            # ax_steady.set_xlabel("Cell radius (μm)")
            # ax_steady.set_ylabel("Steady state concentration (μM)")
            # ax_steady.legend()
            # fig_steady.savefig(
            #     output_folder
            #     / f"steady_state_radius_diffusion_{diffusion}_axisymmetric_{axisymmetric}.{format}"
            # )
            # plt.close(fig_steady)

            # ax_percent.set_xlabel("Cell radius (μm)")
            # ax_percent.set_ylabel("Percent error from analytical solution")
            # ax_percent.legend()
            # fig_percent.savefig(
            #     output_folder
            #     / f"percent_error_radius_diffusion_{diffusion}_axisymmetric_{axisymmetric}.{format}"
            # )
            # plt.close(fig_percent)


def plot_error_different_refinements(all_results, output_folder, format):
    radii = sorted({r.radius for r in all_results})
    diffusions = sorted({r.diffusion for r in all_results})

    for diffusion in diffusions:
        for axisymmetric in [True, False]:

            fig, ax = plt.subplots()
            fig_l2, ax_l2 = plt.subplots()
            fig_time, ax_time = plt.subplots()
            for radius in radii:
                print(radius)
                results = list(
                    sorted(
                        filter(
                            lambda d: np.isclose(d.radius, radius)
                            and np.isclose(d.dt, 0.01)
                            and np.isclose(d.diffusion, diffusion)
                            and (d.axisymmetric is axisymmetric),
                            all_results,
                        ),
                        key=lambda d: d.refinement,
                    )
                )
                refinements = np.array([d.refinement for d in results])
                cA = analytical_solution(radius, D=diffusion)
                ss_vec = np.array([d.ss[-1] for d in results])
                percentError = 100 * np.abs(ss_vec - cA) / cA
                ax.plot(refinements, percentError, marker="o", label=radius)
                ax.set_xticks(refinements)

                l2 = np.array([d.l2 for d in results])
                ax_l2.semilogy(refinements, l2, marker="o", label=radius)
                ax_l2.set_xticks(refinements)

                total_run_time = [
                    float(
                        result.timings[
                            result.timings["name"] == "phosphorylation-example"
                        ].iloc[0]["wall tot"]
                    )
                    for result in results
                ]
                ax_time.semilogy(refinements, total_run_time, marker="o", label=radius)
                ax_time.set_xticks(refinements)

            ax.legend(title="Radius")
            ax.set_xlabel("Refinement")
            ax.set_ylabel("Percent error from analytical solution")
            fig.savefig(
                output_folder
                / f"percent_error_refinement_diffusion_{diffusion}_axisymmetric_{axisymmetric}.{format}"
            )
            plt.close(fig)

            ax_l2.set_xlabel("Refinement")
            ax_l2.legend(title="Radius")
            ax_l2.set_ylabel("$ \| u_e - u \|^2$")
            fig_l2.savefig(
                (
                    output_folder
                    / f"error_refinement_l2_diffusion_{diffusion}_axisymmetric_{axisymmetric}.{format}"
                )
            )
            plt.close(fig_l2)

            ax_time.legend(title="Radius")
            ax_time.set_xlabel("Refinement")
            ax_time.set_ylabel("Total run time [s]")
            fig_time.savefig(
                (
                    output_folder
                    / f"total_time_refinement_diffusion_{diffusion}_axisymmetric_{axisymmetric}.{format}"
                )
            )
            plt.close(fig_time)


def plot_error_different_timesteps(all_results, output_folder, format):
    radii = sorted({r.radius for r in all_results})
    diffusions = sorted({r.diffusion for r in all_results})

    for diffusion in diffusions:
        for axisymmetric in [True, False]:
            fig, ax = plt.subplots()
            fig_l2, ax_l2 = plt.subplots()
            fig_time, ax_time = plt.subplots()
            for radius in radii:
                print(radius)
                results = list(
                    sorted(
                        filter(
                            lambda d: np.isclose(d.radius, radius)
                            and d.refinement == 0
                            and np.isclose(d.diffusion, diffusion)
                            and (d.axisymmetric is axisymmetric),
                            all_results,
                        ),
                        key=lambda d: d.dt,
                    )
                )
                dts = np.array([d.dt for d in results])
                cA = analytical_solution(radius, D=diffusion)
                ss_vec = np.array([d.ss[-1] for d in results])
                percentError = 100 * np.abs(ss_vec - cA) / cA
                ax.plot(dts, percentError, marker="o", label=radius)
                ax.set_xticks(dts)

                l2 = np.array([d.l2 for d in results])
                ax_l2.semilogy(dts, l2, marker="o", label=radius)
                ax_l2.set_xticks(dts)

                total_run_time = [
                    float(
                        result.timings[
                            result.timings["name"] == "phosphorylation-example"
                        ].iloc[0]["wall tot"]
                    )
                    for result in results
                ]
                ax_time.semilogy(dts, total_run_time, marker="o", label=radius)
                ax_time.set_xticks(dts)

            ax.legend(title="Radius")
            ax.set_xlabel("Time step [s]")
            ax.set_ylabel("Percent error from analytical solution")
            fig.savefig(
                output_folder
                / f"percent_error_timestep_diffusion_{diffusion}_axisymmetric_{axisymmetric}.{format}"
            )
            plt.close(fig)

            ax_l2.set_xlabel("Time step [s]")
            ax_l2.set_ylabel("$ \| u_e - u \|^2$")
            fig_l2.savefig(
                (
                    output_folder
                    / f"error_timestep_l2_diffusion_{diffusion}_axisymmetric_{axisymmetric}.{format}"
                )
            )
            plt.close(fig_l2)

            ax_time.legend(title="Radius")
            ax_time.set_xlabel("Time step [s]")
            ax_time.set_ylabel("Total run time [s]")
            fig_time.savefig(
                output_folder
                / f"total_time_timestep_diffusion_{diffusion}_axisymmetric_{axisymmetric}.{format}"
            )
            plt.close(fig_time)


def plot_time_step_vs_refinement(all_results, output_folder, format, axisymmetric=False):
    radii = sorted({r.radius for r in all_results if r.ntasks == 1})
    diffusions = sorted({r.diffusion for r in all_results if r.ntasks == 1})
    refinements = sorted({r.refinement for r in all_results if r.ntasks == 1})
    time_steps = sorted({r.dt for r in all_results if r.ntasks == 1})

    lines = []
    labels = []

    fig, ax = plt.subplots(len(radii), len(diffusions), sharex=True, figsize=(10, 10))
    for i, radius in enumerate(radii):
        ax2 = ax[i, -1].twinx()
        ax2.set_ylabel(f"Radius = {radius}")
        ax2.set_yticks([])
        
        for j, diffusion in enumerate(diffusions):
            ax[0, j].set_title(f"D = {diffusion}")
            
            for refinement in refinements:
                results = list(
                        sorted(
                            filter(
                                lambda d: np.isclose(d.radius, radius)
                                and d.refinement == refinement
                                and np.isclose(d.diffusion, diffusion)
                                and (d.axisymmetric is axisymmetric),
                                all_results,
                            ),
                            key=lambda d: d.dt,
                        )
                    )
                
                # print(refinement, [r.hmin for r in results])
               
                l2 = np.array([d.l2 for d in results])
                dts = np.array([d.dt for d in results])
                if len(results) == 0:
                    continue
                ax[i, j].loglog(dts, l2, marker="o", label=f"hmin {results[0].hmin:.3f}, hmax: {results[0].hmax:.3f}")
            
            ax[i, j].legend()
            

    fig.savefig(
        output_folder / f"timestep_vs_refinement.{format}",
        dpi=300,
    )



def plot_avg_aphos_vs_analytic(all_results, output_folder, format, axisymmetric=False):
    radii = sorted({r.radius for r in all_results if r.ntasks == 1})
    diffusions = sorted({r.diffusion for r in all_results if r.ntasks == 1})
    refinements = sorted({r.refinement for r in all_results if r.ntasks == 1})
    time_steps = sorted({r.dt for r in all_results if r.ntasks == 1})
    fig, ax = plt.subplots(len(radii), len(diffusions), sharex=True, figsize=(15, 10))

    
    lines = []
    labels = []
    for i, radius in enumerate(radii):
        ax2 = ax[i, -1].twinx()
        ax2.set_ylabel(f"Radius = {radius}")
        ax2.set_yticks([])
        
        for j, diffusion in enumerate(diffusions):
            ax[0, j].set_title(f"D = {diffusion}")

            cA = analytical_solution(radius, D=diffusion)
            
            l, = ax[i,j].plot(all_results[0].t, np.ones_like(all_results[0].t) * cA, "k--")
            if i == j == 0:
                lines.append(l)
                labels.append("analytical solution (steady state)")
            for k, time_step in enumerate(time_steps):
                results = list(
                        sorted(
                            filter(
                                lambda d: np.isclose(d.radius, radius)
                                and np.isclose(d.dt, time_step)
                                and np.isclose(d.diffusion, diffusion)
                                and (d.axisymmetric is axisymmetric),
                                all_results,
                            ),
                            key=lambda d: d.refinement,
                        )
                    )
                for ls, marker, r in zip(cycle(linestyles), cycle(markers), results):
                    l, = ax[i, j].plot(r.t, r.ss, color=cmap(k), linestyle=ls, marker=marker)
                    if i == j == 0:
                        lines.append(l)
                        labels.append(f"dt={r.dt:.1e}, refinement:{r.refinement}")
             
            

    lgd = fig.legend(lines, labels, loc="center right")
    fig.subplots_adjust(right=0.79)
    fig.savefig(
        output_folder / f"avg_aphos_vs_analytic.{format}",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
        dpi=300,
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
    results_file = output_folder / "results_phosphorylation.json"
    if skip_if_processed and results_file.is_file():
        print(f"Load results from {results_file}")
        all_results = [Data(**r) for r in json.loads(results_file.read_text())]
    else:
        print(f"Gather results from {results_folder}")
        all_results = load_results(Path(results_folder))
        print(f"Save results to {results_file.absolute()}")
        results_file.write_text(
            json.dumps([r.to_json() for r in all_results], indent=4)
        )

    # plot_error_analytical_solution_different_radius(all_results, output_folder, format)
    # plot_error_different_refinements(all_results, output_folder, format)
    # plot_error_different_timesteps(all_results, output_folder, format)
    plot_time_step_vs_refinement(all_results, output_folder, format)
    # plot_avg_aphos_vs_analytic(all_results, output_folder, format)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    phosphorylation_parser_args.add_phosphorylation_postprocess(parser)
    args = vars(parser.parse_args())
    raise SystemExit(main(**args))
