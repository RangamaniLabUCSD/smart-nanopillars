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
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import phosphorylation_parser_args


class Data(NamedTuple):
    t: np.ndarray
    ss: np.ndarray
    l2: np.ndarray
    config: Dict[str, Any]

    @property
    def radius(self):
        return self.config["curRadius"]

    @property
    def dt(self):
        return self.config["time_step"]

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
        }


def load_results(folder):
    data = []
    for result_folder in Path(folder).iterdir():
        if not result_folder.is_dir():
            continue
        if not (result_folder / "avg_Aphos.txt").exists():
            continue
        config = json.loads((result_folder / "config.json").read_text())
        t = np.loadtxt(result_folder / "tvec.txt")
        ss = np.loadtxt(result_folder / "avg_Aphos.txt")
        l2 = np.loadtxt(result_folder / "L2norm.txt")
        data.append(Data(t=t, ss=ss, l2=l2, config=config))
        print(data[-1])

    if len(data) == 0:
        raise RuntimeError(f"No results found in folder {folder}")
    return data

def analytical_solution(radius):
    thieleMod = radius / 1.0
    k_kin = 50
    k_p = 10
    cT = 1
    D = 10
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
    results = list(
        sorted(
            filter(lambda d: d.refinement == 0 and np.isclose(d.dt, 0.01), all_results),
            key=lambda d: d.radius,
        )
    )
    radiusVec = np.array([d.radius for d in results])
    cA = analytical_solution(radiusVec)
    radiusTest = np.logspace(0, 1, 100)
    cA_smooth = analytical_solution(radiusTest)
    ss_vec = np.array([d.ss[-1] for d in results])

    fig, ax = plt.subplots()
    ax.plot(radiusVec, ss_vec, "ro")
    
    ax.plot(radiusTest, cA_smooth)
    ax.set_xlabel("Cell radius (μm)")
    ax.set_ylabel("Steady state concentration (μM)")
    ax.legend(("SMART simulation", "Analytical solution"))
    fig.savefig((output_folder / "steady_state_radius").with_suffix(f".{format}"))
    plt.close(fig)

    # We compare the SMART results to the analytical solution, requiring that the steady state concentration
    # in simulations deviates less than 1% from the known analytical value.

    # quantify percent error
   
    percentError = 100 * np.abs(ss_vec - cA) / cA
    fig, ax = plt.subplots()
    ax.plot(radiusVec, percentError)
    ax.set_xlabel("Cell radius (μm)")
    ax.set_ylabel("Percent error from analytical solution")
    fig.savefig((output_folder / "percent_error_radius").with_suffix(f".{format}"))
    plt.close(fig)

    # assert all(
    #     percentError < 1
    # ), f"Example 2 results deviate {max(percentError):.3f}% from the analytical solution"
    rmse = np.sqrt(np.mean(percentError**2))
    print(f"RMSE with respect to analytical solution = {rmse:.3f}%")

    # Plot l2 erro for differnt radius
    l2 = np.array([d.l2 for d in results])
    fig, ax = plt.subplots()
    ax.semilogy(radiusVec, l2)
    ax.set_xlabel("Cell radius (μm)")
    ax.set_ylabel("$ \| u_e - u \|^2$")
    ax.set_title("Percent $\ell^2$ error from analytical solution")
    fig.savefig((output_folder / "percent_error_radius_l2.png").with_suffix(f".{format}"))
    plt.close(fig)
   


def plot_error_different_refinements(all_results, output_folder, format):
    radii = sorted({r.radius for r in all_results})
    fig, ax = plt.subplots()
    fig_l2, ax_l2 = plt.subplots()
    for radius in radii:
        print(radius)
        results = list(
            sorted(
                filter(lambda d: np.isclose(d.radius, radius) and np.isclose(d.dt, 0.01), all_results),
                key=lambda d: d.refinement,
            )
        )
        refinements = np.array([d.refinement for d in results])
        cA = analytical_solution(radius)
        ss_vec = np.array([d.ss[-1] for d in results])
        percentError = 100 * np.abs(ss_vec - cA) / cA
        ax.plot(refinements, percentError, marker="o", label=radius)
        ax.set_xticks(refinements)
        
        l2 = np.array([d.l2 for d in results])
        ax_l2.semilogy(refinements, l2, marker="o", label=radius)
        ax_l2.set_xticks(refinements)


    ax.legend(title="Radius")
    ax.set_xlabel("Refinement")
    ax.set_ylabel("Percent error from analytical solution")
    fig.savefig((output_folder / "percent_error_refinement").with_suffix(f".{format}"))
    plt.close(fig)

    ax_l2.set_xlabel("Refinement")
    ax_l2.set_ylabel("$ \| u_e - u \|^2$")
    ax_l2.set_title("Percent $\ell^2$ error from analytical solution")
    fig_l2.savefig((output_folder / "percent_error_refinement_l2.png").with_suffix(f".{format}"))
    plt.close(fig_l2)


def plot_error_different_timesteps(all_results, output_folder, format):
    radii = sorted({r.radius for r in all_results})
    fig, ax = plt.subplots()
    fig_l2, ax_l2 = plt.subplots()
    for radius in radii:
        print(radius)
        results = list(
            sorted(
                filter(lambda d: np.isclose(d.radius, radius) and d.refinement == 0, all_results),
                key=lambda d: d.dt,
            )
        )
        dts = np.array([d.dt for d in results])
        cA = analytical_solution(radius)
        ss_vec = np.array([d.ss[-1] for d in results])
        percentError = 100 * np.abs(ss_vec - cA) / cA
        ax.plot(dts, percentError, marker="o", label=radius)
        ax.set_xticks(dts)

        l2 = np.array([d.l2 for d in results])
        ax_l2.semilogy(dts, l2, marker="o", label=radius)
        ax_l2.set_xticks(dts)

    ax.legend(title="Radius")
    ax.set_xlabel("Time step [s]")
    ax.set_ylabel("Percent error from analytical solution")
    fig.savefig((output_folder / "percent_error_timestep").with_suffix(f".{format}"))
    plt.close(fig)

    ax_l2.set_xlabel("Time step [s]")
    ax_l2.set_ylabel("$ \| u_e - u \|^2$")
    ax_l2.set_title("Percent $\ell^2$ error from analytical solution")
    fig_l2.savefig((output_folder / "percent_error_timestep_l2.png").with_suffix(f".{format}"))
    plt.close(fig_l2)

   
def main(results_folder: Path, output_folder: Path, format: str = "png", skip_if_processed: bool = False, use_tex: bool = False) -> int:
    plt.rcParams['text.usetex'] = use_tex

    output_folder.mkdir(exist_ok=True, parents=True)
    results_file = output_folder / "results_phosphorylation.json"
    if skip_if_processed and results_file.is_file():
        print(f"Load results from {results_file}")
        all_results = [Data(**r) for r in json.loads(results_file.read_text())]
    else:
        print(f"Gather results from {results_folder}")
        all_results = load_results(Path(results_folder))
        print(f"Save results to {results_file.absolute()}")
        (output_folder / "results_phosphorylation.json").write_text(json.dumps([r.to_json() for r in all_results], indent=4))

    plot_error_analytical_solution_different_radius(all_results, output_folder, format)
    plot_error_different_refinements(all_results, output_folder, format)
    plot_error_different_timesteps(all_results, output_folder, format)
    return 0

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    phosphorylation_parser_args.add_phosphorylation_postprocess(parser)
    args = vars(parser.parse_args())
    raise SystemExit(main(**args))
