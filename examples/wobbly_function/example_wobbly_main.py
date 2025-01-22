"""This module sets up and executes an optimization experiment using the Yotse
framework, specifically designed to optimize a 'wobbly function'.

It leverages genetic algorithms for the optimization process, handles file management,
and cleans up after execution, showcasing a very basic use-case.
"""

import os
import shutil
from typing import Any

import numpy as np
import scipy

from yotse.execution import Executor
from yotse.pre import Experiment
from yotse.pre import OptimizationInfo
from yotse.pre import Parameter
from yotse.pre import SystemSetup


def wobbly_pre() -> Experiment:
    """Configures and returns an experiment setup for optimizing a wobbly function.

    Returns
    -------
    Experiment
        The configured Experiment object for the wobbly function optimization.
    """
    wobbly_experiment = Experiment(
        experiment_name="wobbly_example",
        system_setup=SystemSetup(
            source_directory=os.getcwd(),
            program_name="wobbly_function.py",
            command_line_arguments={
                "--filebasename": "wobbly_example",
                # '--resume': '.qcgpjm-service-david-latitude7430.6070'
            },
            analysis_script="analyse_function_output.py",
            executor="python",
            # files_needed=["*.py"] # todo not implemented
        ),
        parameters=[
            Parameter(
                name="x",
                param_range=[-4, 4],
                number_points=4,
                distribution="uniform",
                constraints=None,
                weights=None,  # todo not implemented
                parameter_active=True,
                param_type="continuous",
            ),
            Parameter(
                name="y",
                param_range=[-3, 3],
                number_points=4,
                distribution="uniform",
                constraints={"low": -4, "high": 4, "step": 0.001},
                weights=None,
                parameter_active=True,
                param_type="continuous",
            ),
        ],
        opt_info_list=[
            OptimizationInfo(
                name="GA",
                blackbox_optimization=True,
                opt_parameters={
                    "num_generations": 5,  # number of iterations of the algorithm
                    # "num_points": 10,            # number of points per param to re-create , now determined by initial
                    "num_parents_mating": 5,
                    "mutation_probability": 0.2,
                    "refinement_factors": [0.5, 0.5],
                    "logging_level": 1,
                },
                is_active=True,
            ),
            OptimizationInfo(
                name="scipy",
                blackbox_optimization=False,
                opt_parameters={
                    "method": "Nelder-Mead",
                    "bounds": None,
                    "options": {"maxiter": 1000, "disp": True},
                },
                is_active=False,
            ),
        ],
    )
    return wobbly_experiment


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


def scipy_callback(intermediate_result: scipy.optimize.OptimizeResult) -> None:
    """SciPy callback for printing intermediate result."""
    print(f"Current x: {intermediate_result.x}")
    print(f"Current fun value : {intermediate_result.fun}")
    return


def remove_files_after_run() -> None:
    """Removes output directories and QCG temporary files after the optimization run."""
    # remove files and directories
    shutil.rmtree("output")
    dirs = [f for f in os.listdir(os.getcwd()) if (f.startswith(".qcg"))]
    for d in dirs:
        shutil.rmtree(os.path.join(os.getcwd(), d))


def main() -> None:
    """Main execution function that initializes the experiment and executes multiple
    different optimization steps consecutively.

    First we perform a blackbox optimization using Genetic Algorithm that we then follow
    up by a whitebox optimization using Scipy Optimization. This is meant to demonstrate
    how to explore an unknown optimization problem and once an extrema has been found,
    exploit this using knowledge about the shape of the function around this point.
    """
    print("\033[93m --- Executing Wobbly-Main Example. --- \033[0m")
    experiment = wobbly_pre()
    wobbly_example = Executor(experiment=experiment)

    for i in range(wobbly_example.optimizer.optimization_algorithm.max_iterations):
        # todo : the grid based point generation is still somehow bugged
        # wobbly_example.run(step=i, evolutionary_point_generation=False)
        wobbly_example.run(step_number=i, evolutionary_point_generation=True)

    solution = wobbly_example.optimizer.suggest_best_solution()
    print("Solution after GA: ", solution)
    # matplotlib.use('Qt5Agg')
    # wobbly_example.optimization_alg.ga_instance.plot_new_solution_rate()
    # wobbly_example.optimization_alg.ga_instance.plot_fitness()

    # add remaining params to second optimization
    experiment.opt_info_list[1].opt_parameters["fun"] = ackley_function_2d
    experiment.opt_info_list[1].opt_parameters["callback"] = scipy_callback
    experiment.opt_info_list[1].opt_parameters["x0"] = np.array(solution[0])
    print(experiment.opt_info_list[1].opt_parameters["x0"])

    wobbly_example.next_optimization()

    wobbly_example.run()

    solution = wobbly_example.optimizer.suggest_best_solution()
    print("Solution after SciPy: ", solution)

    remove_files_after_run()


if __name__ == "__main__":
    main()
