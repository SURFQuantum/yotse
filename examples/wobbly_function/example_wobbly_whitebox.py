import random
from typing import Any

import numpy as np
import scipy

from examples.wobbly_function.example_wobbly_main import wobbly_pre
from yotse.execution import Executor
from yotse.pre import OptimizationInfo


def ackley_function_2d(x: np.ndarray, *args: Any) -> float:
    """Returns function value of the 2d ackley function."""
    if len(x) != 2:
        raise ValueError("Input array x must have shape (2,).")

    x, y = x[0], x[1]
    f = float(
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        + np.e
        + 20
    )
    return f


def main() -> None:
    print("\033[93m --- Executing Wobbly-Whitebox Example. --- \033[0m")

    whitebox_experiment = wobbly_pre()

    # pick random initial guess
    initial_guess = np.array(
        [
            random.choice(whitebox_experiment.parameters[0].data_points),
            random.choice(whitebox_experiment.parameters[1].data_points),
        ]
    )

    def scipy_callback(intermediate_result: scipy.optimize.OptimizeResult) -> None:
        print("Current x", intermediate_result.x)
        print("Current fun", intermediate_result.fun)
        return

    whitebox_opt = OptimizationInfo(
        name="scipy",
        blackbox_optimization=False,
        opt_parameters={
            "fun": ackley_function_2d,
            "x0": initial_guess,
            "method": "Nelder-Mead",
            "bounds": None,
            "options": {"maxiter": 1000, "disp": True},
            "callback": scipy_callback,
        },
        is_active=True,
    )
    # deactivate other optimization
    whitebox_experiment.optimization_information_list[0].is_active = False
    # append new active whitebox optimization
    whitebox_experiment.optimization_information_list.append(whitebox_opt)

    wobbly_whitebox = Executor(experiment=whitebox_experiment)

    wobbly_whitebox.run()

    solution = wobbly_whitebox.optimizer.suggest_best_solution()
    print("Solution: ", solution)
    # matplotlib.use('Qt5Agg')
    # wobbly_example.optimization_alg.ga_instance.plot_new_solution_rate()
    # wobbly_example.optimization_alg.ga_instance.plot_fitness()
    # remove_files_after_run()


if __name__ == "__main__":
    main()
