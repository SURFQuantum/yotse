"""Example script for execution of a wobbly_function.py experiment."""
import os
import shutil

from yotse.execution import Executor
from yotse.pre import Experiment
from yotse.pre import OptimizationInfo
from yotse.pre import Parameter
from yotse.pre import SystemSetup

# import matplotlib


def wobbly_pre() -> Experiment:
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
                opt_parameters={
                    "num_generations": 10,  # number of iterations of the algorithm
                    # "num_points": 10,            # number of points per param to re-create , now determined by initial
                    "num_parents_mating": 5,
                    "mutation_probability": 0.2,
                    "refinement_factors": [0.5, 0.5],
                    "logging_level": 1,
                },
                is_active=True,
            )
        ],
    )
    return wobbly_experiment


def remove_files_after_run() -> None:
    # remove files and directories
    shutil.rmtree("output")
    dirs = [f for f in os.listdir(os.getcwd()) if (f.startswith(".qcg"))]
    for d in dirs:
        shutil.rmtree(os.path.join(os.getcwd(), d))


def main() -> None:
    print(" --- Running Wobbly-Main Example. --- ")
    experiment = wobbly_pre()
    wobbly_example = Executor(experiment=experiment)

    for i in range(
        wobbly_example.optimizer.optimization_algorithm.optimization_instance.generations_completed,
        experiment.optimization_information_list[0].parameters["num_generations"],
    ):
        assert (
            wobbly_example.optimizer.optimization_algorithm.optimization_instance.generations_completed
            == i
        )
        # todo : the grid based point generation is still somehow bugged
        # wobbly_example.run(step=i, evolutionary_point_generation=False)
        wobbly_example.run(step_number=i, evolutionary_point_generation=True)

    solution = wobbly_example.optimizer.suggest_best_solution()
    print("Solution: ", solution)
    # matplotlib.use('Qt5Agg')
    # wobbly_example.optimization_alg.ga_instance.plot_new_solution_rate()
    # wobbly_example.optimization_alg.ga_instance.plot_fitness()
    remove_files_after_run()


if __name__ == "__main__":
    main()
