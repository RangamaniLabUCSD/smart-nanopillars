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
import sys
import matplotlib.pyplot as plt
import numpy as np
import json


class Data(NamedTuple):
    t: np.ndarray
    ss: np.ndarray
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
        data.append(Data(t, ss, config))
        print(data[-1])

    if len(data) == 0:
        raise RuntimeError(f"No results found in folder {folder}")
    return data


def main(results_folder):
    all_results = load_results(Path(results_folder))
    results = list(
        sorted(
            filter(lambda d: d.refinement == 0 and np.isclose(d.dt, 0.01), all_results),
            key=lambda d: d.radius,
        )
    )
    radiusVec = np.array([d.radius for d in results])
    ss_vec = np.array([d.ss[-1] for d in results])

    fig, ax = plt.subplots()
    ax.plot(radiusVec, ss_vec, "ro")
    radiusTest = np.logspace(0, 1, 100)
    thieleMod = radiusTest / 1.0
    k_kin = 50
    k_p = 10
    cT = 1
    D = 10
    C1 = (
        k_kin
        * cT
        * radiusTest**2
        / (
            (3 * D * (np.sqrt(k_p / D) - (1 / radiusTest)) + k_kin * radiusTest)
            * np.exp(thieleMod)
            + (3 * D * (np.sqrt(k_p / D) + (1 / radiusTest)) - k_kin * radiusTest)
            * np.exp(-thieleMod)
        )
    )
    cA = (6 * C1 / radiusTest) * (
        np.cosh(thieleMod) / thieleMod - np.sinh(thieleMod) / thieleMod**2
    )
    ax.plot(radiusTest, cA)
    ax.set_xlabel("Cell radius (μm)")
    ax.set_ylabel("Steady state concentration (μM)")
    ax.legend(("SMART simulation", "Analytical solution"))
    fig.savefig("steady_state.png")

    # We compare the SMART results to the analytical solution, requiring that the steady state concentration
    # in simulations deviates less than 1% from the known analytical value.

    # quantify percent error
    thieleMod = radiusVec / 1.0
    k_kin = 50
    k_p = 10
    cT = 1
    D = 10
    C1 = (
        k_kin
        * cT
        * radiusVec**2
        / (
            (3 * D * (np.sqrt(k_p / D) - (1 / radiusVec)) + k_kin * radiusVec)
            * np.exp(thieleMod)
            + (3 * D * (np.sqrt(k_p / D) + (1 / radiusVec)) - k_kin * radiusVec)
            * np.exp(-thieleMod)
        )
    )
    cA = (6 * C1 / radiusVec) * (
        np.cosh(thieleMod) / thieleMod - np.sinh(thieleMod) / thieleMod**2
    )
    percentError = 100 * np.abs(ss_vec - cA) / cA
    fig, ax = plt.subplots()
    ax.plot(radiusVec, percentError)
    ax.set_xlabel("Cell radius (μm)")
    ax.set_ylabel("Percent error from analytical solution")
    fig.savefig("percent_error.png")
    assert all(
        percentError < 1
    ), f"Example 2 results deviate {max(percentError):.3f}% from the analytical solution"
    rmse = np.sqrt(np.mean(percentError**2))
    print(f"RMSE with respect to analytical solution = {rmse:.3f}%")


if __name__ == "__main__":
    try:
        results_folder = sys.argv[1]
    except IndexError:
        print("Please provide the path to the results folder as an argument")
        raise SystemExit(1)

    raise SystemExit(main(results_folder))
