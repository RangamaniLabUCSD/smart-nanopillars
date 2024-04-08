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
import rlcompleter
from itertools import cycle
import json
import phosphorylation_parser_args

ntasks_pattern = re.compile("ntasks: (?P<n>\d+)")
cmap = plt.get_cmap("tab10")
linestyles = ["-", "--", "-.", ":"]
markers = ["", ".", "o"]


class Data(NamedTuple):
    t: np.ndarray
    ss: np.ndarray
    l2: np.ndarray
    config: Dict[str, Any]
    timings_: Dict[int, Dict[str, Any]]
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
        return {k: pd.DataFrame(t) for k, t in self.timings_.items()}

    @property
    def min_run_time(self) -> float:
        """Return the sum of the run time of all ranks."""
        return min(self.run_time_dict.values())
    
    @property
    def mean_run_time(self) -> float:
        """Return the sum of the run time of all ranks."""
        return np.mean(list(self.run_time_dict.values()))

    @property
    def max_run_time(self) -> float:
        """Return the maximum run time of all ranks."""
        return max(self.run_time_dict.values())

    @property
    def run_time_dict(self) -> Dict[int, float]:
        return {
            k: float(v[v["name"] == "phosphorylation-example"].iloc[0]["wall tot"])
            for k, v in self.timings.items()
        }

    @property
    def hmin(self) -> float:
        return self.config.get("hmin", -1.0)

    @property
    def hmax(self) -> float:
        return self.config.get("hmax", -1.0)

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

        timings = {
            int(f.stem.lstrip("timings_rank")): parse_timings(f.read_text())
            for f in result_folder.glob("timings_rank*")
        }
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
    dts = sorted({r.dt for r in all_results})
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
                fig_steady, ax_steady = plt.subplots()
                fig_percent, ax_percent = plt.subplots()
                fig_l2, ax_l2 = plt.subplots()
                fig_time, ax_time = plt.subplots()
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
                            key=lambda d: d.dt,
                        )
                    )
                    radiusVec = np.array([d.radius for d in results])
                    cA = analytical_solution(radiusVec, D=diffusion)
                    ss_vec = np.array([d.ss[-1] for d in results])
                    percentError = 100 * np.abs(ss_vec - cA) / cA
                    l2 = np.array([d.l2 for d in results])
                    total_run_time = [result.max_run_time for result in results]

                ax_steady.plot(
                    radiusVec,
                    ss_vec,
                    linestyle=":",
                    marker="o",
                    label=f"SMART simulation (refinement: {refinement})",
                )
                ax_percent.plot(
                    radiusVec, percentError, label=f"refinement {refinement}"
                )
                ax_l2.semilogy(radiusVec, l2, label=f"refinement {refinement}")
                rmse = np.sqrt(np.mean(percentError**2))
                print(
                    f"RMSE with respect to analytical solution = {rmse:.3f}% for refinement {refinement}, D: {diffusion}, axisymetric: {axisymmetric}"
                )
                ax_time.plot(radiusVec, total_run_time, marker="o")

            radiusTest = np.logspace(0, 1, 100)
            cA_smooth = analytical_solution(radiusTest, D=diffusion)
            ax_steady.plot(radiusTest, cA_smooth, label="Analytical solution")

            ax_l2.set_xlabel("Cell radius (μm)")
            ax_l2.set_ylabel("$ \| u_e - u \|^2$")
            ax_l2.set_title("$\ell^2$ error from analytical solution")
            ax_l2.legend()
            fig_l2.savefig(
                (
                    output_folder
                    / f"error_radius_l2_diffusion_{diffusion}_axisymmetric_{axisymmetric}.{format}"
                )
            )
            plt.close(fig_l2)

            ax_time.set_xlabel("Cell radius (μm)")
            ax_time.set_ylabel("Total run time [s]")
            ax_time.legend()
            fig_time.savefig(
                output_folder
                / f"total_time_radius_diffusion_{diffusion}_axisymmetric_{axisymmetric}.{format}"
            )
            plt.close(fig_time)

            ax_steady.set_xlabel("Cell radius (μm)")
            ax_steady.set_ylabel("Steady state concentration (μM)")
            ax_steady.legend()
            fig_steady.savefig(
                output_folder
                / f"steady_state_radius_diffusion_{diffusion}_axisymmetric_{axisymmetric}.{format}"
            )
            plt.close(fig_steady)

            ax_percent.set_xlabel("Cell radius (μm)")
            ax_percent.set_ylabel("Percent error from analytical solution")
            ax_percent.legend()
            fig_percent.savefig(
                output_folder
                / f"percent_error_radius_diffusion_{diffusion}_axisymmetric_{axisymmetric}.{format}"
            )
            plt.close(fig_percent)


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

                total_run_time = [result.max_run_time for result in results]
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

                total_run_time = [result.max_run_time for result in results]
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


def plot_time_step_vs_refinement(
    all_results, output_folder, format, axisymmetric=False
):
    radii = sorted({r.radius for r in all_results if r.ntasks == 1})
    diffusions = sorted({r.diffusion for r in all_results if r.ntasks == 1})
    refinements = sorted({r.refinement for r in all_results if r.ntasks == 1})
    time_steps = sorted({r.dt for r in all_results if r.ntasks == 1})
    x = np.arange(len(time_steps))
    width = 0.9 / len(refinements)
    rates = []

    fig, ax = plt.subplots(len(radii), len(diffusions), sharex=True, figsize=(10, 10))
    fig_t, ax_t = plt.subplots(
        len(radii), len(diffusions), sharex=True, figsize=(10, 10)
    )
    for i, radius in enumerate(radii):
        for axi in [ax, ax_t]:
            ax2 = axi[i, -1].twinx()
            ax2.set_ylabel(f"Radius = {radius}")
            ax2.set_yticks([])

        for j, diffusion in enumerate(diffusions):
            for axi in [ax, ax_t]:
                axi[0, j].set_title(f"D = {diffusion}")

            for k, refinement in enumerate(refinements):
                results = list(
                    sorted(
                        filter(
                            lambda d: np.isclose(d.radius, radius)
                            and d.refinement == refinement
                            and np.isclose(d.diffusion, diffusion)
                            and d.ntasks == 1
                            and (d.axisymmetric is axisymmetric),
                            all_results,
                        ),
                        key=lambda d: d.dt,
                    )
                )
                if len(results) == 0:
                    continue
       
                dts = np.array([d.dt for d in results])
                l2 = np.array([d.l2 for d in results])
                timings = np.array([d.max_run_time for d in results])

                for r in results:
                    rates.append(
                        {
                            "radius": radius,
                            "diffusion": diffusion,
                            "refinement": refinement,
                            "hmin": r.hmin,
                            "hmax": r.hmax,
                            "dt": r.dt,
                            "l2": float(r.l2),
                        }
                    )

                
                ax[i, j].loglog(
                    dts,
                    l2,
                    marker="o",
                    label=f"hmin {results[0].hmin:.3f}, hmax: {results[0].hmax:.3f}",
                )
                ax_t[i, j].bar(
                    x[np.isin(time_steps, dts)] + k * width,
                    timings,
                    width,
                    label=f"refinement {refinement}",
                )
            ax[i, j].legend()
            ax_t[i, j].legend()
            ax[i, j].set_xlabel("Time step [s]")
            ax[i, j].set_ylabel(r"$ \| u_e - u \|^2$")
            ax_t[i, j].set_xlabel("Time step [s]")
            ax_t[i, j].set_ylabel("Total run time [s]")
            ax_t[i, j].set_xticks(x + width * len(refinements) / 2)
            ax_t[i, j].set_xticklabels([f"{t:.1e}" for t in time_steps], rotation=45)
            ax_t[i, j].set_yscale("log")

    fig.savefig(
        output_folder / f"timestep_vs_refinement.{format}",
        dpi=300,
    )
    fig_t.savefig(
        output_folder / f"total_time_vs_refinement.{format}",
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

            (l,) = ax[i, j].plot(
                all_results[0].t, np.ones_like(all_results[0].t) * cA, "k--"
            )
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
                    (l,) = ax[i, j].plot(
                        r.t, r.ss, color=cmap(k), linestyle=ls, marker=marker
                    )
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


def get_convergence_rates(all_results, output_folder, axisymmetric=False):
    radii = sorted({r.radius for r in all_results if r.ntasks == 1})
    diffusions = sorted({r.diffusion for r in all_results if r.ntasks == 1})
    refinements = sorted({r.refinement for r in all_results if r.ntasks == 1})
    time_steps = sorted({r.dt for r in all_results if r.ntasks == 1})

    rates = []

    for i, radius in enumerate(radii):

        for j, diffusion in enumerate(diffusions):

            for k, refinement in enumerate(refinements):
                results = list(
                    sorted(
                        filter(
                            lambda d: np.isclose(d.radius, radius)
                            and d.refinement == refinement
                            and np.isclose(d.diffusion, diffusion)
                            and d.ntasks == 1
                            and (d.axisymmetric is axisymmetric),
                            all_results,
                        ),
                        key=lambda d: d.dt,
                    )
                )

                for r in results:
                    rates.append(
                        {
                            "radius": radius,
                            "diffusion": diffusion,
                            "refinement": refinement,
                            "hmin": r.hmin,
                            "hmax": r.hmax,
                            "dt": r.dt,
                            "l2": float(r.l2),
                        }
                    )

    rates_df = pd.DataFrame(rates)
    rates_spatial = []
    rates_temporal = []

    for radius in radii:
        rates_radius = rates_df[rates_df["radius"] == radius]
        for diffusion in diffusions:
            rates_diffusion = rates_radius[rates_radius["diffusion"] == diffusion]

            # Spatial convergence rates
            for dt in time_steps:
                rates_dt = rates_diffusion[rates_diffusion["dt"] == dt].sort_values(
                    by="hmin", ascending=False
                )
                h = rates_dt.hmin[1:].values
                h_ = rates_dt.hmin[:-1].values

                e = rates_dt.l2[1:].values
                e_ = rates_dt.l2[:-1].values

                r = np.log(e / e_) / np.log(h / h_)
                for i in range(1, len(r)):
                    rates_spatial.append(
                        {
                            "radius": radius,
                            "diffusion": diffusion,
                            "hmin_i": h[i],
                            "hmin_i-1": h_[i],
                            "l2_i": e[i],
                            "l2_i-1": e_[i],
                            "dt": dt,
                            "rate": r[i],
                        }
                    )

            # Temporal convergence rates
            for refinement in refinements:
                rates_ref = rates_diffusion[
                    rates_diffusion["refinement"] == refinement
                ].sort_values(by="dt", ascending=False)
                dt = rates_ref.dt.values[1:]
                dt_ = rates_ref.dt.values[:-1]

                e = rates_ref.l2.values[1:]
                e_ = rates_ref.l2.values[:-1]

                r = np.log(e / e_) / np.log(dt / dt_)
                for i in range(1, len(r)):
                    rates_temporal.append(
                        {
                            "radius": radius,
                            "diffusion": diffusion,
                            "dt_i": dt[i],
                            "dt_i-1": dt_[i],
                            "l2_i": e[i],
                            "l2_i-1": e_[i],
                            "refinement": refinement,
                            "hmin": rates_ref.hmin.values[i],
                            "rate": r[i],
                        }
                    )

    rates_spatial_df = pd.DataFrame(rates_spatial)
    rates_temporal_df = pd.DataFrame(rates_temporal)

    rates_spatial_df.to_csv(output_folder / "rates_spatial.csv", index=False)
    rates_temporal_df.to_csv(output_folder / "rates_temporal.csv", index=False)


def plot_scalability(all_results, output_folder, format):

    all_ntasks = sorted({r.ntasks for r in all_results})
    radius = 1.0
    diffusion = 10.0
    dt = 0.01
    axisymmetric = False
    x = np.arange(len(all_ntasks))
    width = 0.9 / 3
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(12, 10))
    for i, refinement in enumerate([0, 3]):
        results = list(
            sorted(
                filter(
                    lambda d: np.isclose(d.radius, radius)
                    and d.refinement == refinement
                    and np.isclose(d.diffusion, diffusion)
                    and np.isclose(d.dt, dt)
                    and d.axisymmetric is axisymmetric,
                    all_results,
                ),
                key=lambda d: d.ntasks,
            )
        )

        l2 = np.array([d.l2 for d in results])
        ntasks = np.array([d.ntasks for d in results])
        # sys_run_time = np.array([d.system_run_time for d in results])
        # user_run_time = np.array([d.user_run_time for d in results])
        mean_run_time = np.array([d.mean_run_time for d in results])
        min_run_time = np.array([d.min_run_time for d in results])
        max_run_time = np.array([d.max_run_time for d in results])
        # run_time_dict = [list(r.run_time_dict.values()) for r in results]
        # breakpoint()
        # run_time_dict
        # ax[0, i].violinplot(run_time_dict)#, labels=[f"{t:d}" for t in ntasks])
        ax[0, i].bar(x[np.isin(ntasks, all_ntasks)], max_run_time,  width, label="Max run time")
        ax[0, i].bar(x[np.isin(ntasks, all_ntasks)] + width, mean_run_time, width, label="Mean run time")
        ax[0, i].bar(x[np.isin(ntasks, all_ntasks)] + 2 * width, mean_run_time, width, label="Min run time")
        ax[0, i].set_xticks(x + width)
        ax[0, i].set_xticklabels([f"{t:d}" for t in all_ntasks], rotation=45)
        ax[0, i].set_yscale("log")
        ax[0, i].set_xlabel("Number of tasks")
        ax[1, i].bar(x[np.isin(ntasks, all_ntasks)], l2, width)
        ax[1, i].set_xticks(x)
        ax[1, i].set_xticklabels([f"{t:d}" for t in all_ntasks], rotation=45)
        ax[1, i].set_yscale("log")
        ax[1, i].set_xlabel("Number of tasks")
        ax[1, i].set_ylim(1e-10, 1e-2)    
        ax[0, i].set_title(f"Refinement {refinement}")

        if i == 0:
            ax[0, i].set_ylabel("Run time [s]")
            ax[1, i].set_ylabel(r"$ \| u_e - u \|^2$")
            ax[0, i].legend()
        

    fig.savefig(
        output_folder / f"scalability.{format}",
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
        

    get_convergence_rates(all_results, output_folder)
    plot_time_step_vs_refinement(all_results, output_folder, format)
    plot_avg_aphos_vs_analytic(all_results, output_folder, format)

    plot_scalability(all_results, output_folder, format)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    phosphorylation_parser_args.add_phosphorylation_postprocess(parser)
    args = vars(parser.parse_args())
    raise SystemExit(main(**args))
