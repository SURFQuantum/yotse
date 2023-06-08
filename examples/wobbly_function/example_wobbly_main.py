"""Example script for execution of a wobbly_function.py experiment."""
import os
import matplotlib
import shutil
from qiaopt.pre import Experiment, SystemSetup, Parameter, OptimizationInfo
from qiaopt.run import Executor


def wobbly_pre():
    wobbly_experiment = Experiment(
        experiment_name="wobbly_example",
        system_setup=SystemSetup(source_directory=os.getcwd(),
                                 program_name='wobbly_function.py',
                                 command_line_arguments={"--filebasename": 'wobbly_example'},
                                 analysis_script="analyse_function_output.py",
                                 executor="python",
                                 # files_needed=["*.py"] # todo not implemented
                                 ),
        parameters=[
            Parameter(
                name="x",
                param_range=[-4, 4],
                number_points=10,
                distribution="uniform",
                constraints=None,
                weights=None,   # todo not implemented
                parameter_active=True,
                param_type="continuous"
            ),
            Parameter(
                name="y",
                param_range=[-3, 3],
                number_points=10,
                distribution="uniform",
                constraints={'low': -4, 'high': 4, 'step': .001},
                weights=None,
                parameter_active=True,
                param_type="continuous"
            )
        ],
        opt_info_list=[
            OptimizationInfo(
                name="GA",
                opt_parameters={
                    "num_generations": 10,     # number of iterations of the algorithm
                    # "num_points": 10,            # number of points per param to re-create , now determined by initial
                    "num_parents_mating": 5,
                    "mutation_probability": .2,
                    "refinement_factors": [.5, .5],
                    "logging_level": 1,
                },
                is_active=True)]
    )
    return wobbly_experiment


def remove_files_after_run():
    # remove files and directories
    shutil.rmtree('output')
    dirs = [f for f in os.listdir(os.getcwd()) if (f.startswith(".qcg"))]
    for d in dirs:
        shutil.rmtree(os.path.join(os.getcwd(), d))


def main():
    def cost_function(f):
        return f

    experiment = wobbly_pre()
    experiment.cost_function = cost_function
    wobbly_example = Executor(experiment=experiment)

    for i in range(experiment.optimization_information_list[0].parameters["num_generations"]):
        assert wobbly_example.optimization_alg.ga_instance.generations_completed == i   # sanity check
        # todo : the grid based point generation is still somehow bugged
        # wobbly_example.run(step=i, evolutionary_point_generation=False)
        wobbly_example.run(step=i, evolutionary_point_generation=True)

    solution = wobbly_example.suggest_best_solution()
    print("Solution: ", solution)
    # matplotlib.use('Qt5Agg')
    # wobbly_example.optimization_alg.ga_instance.plot_new_solution_rate()
    # wobbly_example.optimization_alg.ga_instance.plot_fitness()
    remove_files_after_run()


if __name__ == "__main__":
    main()
