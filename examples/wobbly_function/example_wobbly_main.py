"""Example script for execution of a wobbly_function.py experiment."""
import os
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
                                 files_needed=["*.py"]),
        parameters=[
            Parameter(
                name="x",
                parameter_range=[-4, 4],
                number_points=10,
                distribution="uniform",
                constraints=[],
                weights=None,
                parameter_active=True,
                data_type="continuous"
            ),
            Parameter(
                name="y",
                parameter_range=[-3, 3],
                number_points=10,
                distribution="uniform",
                constraints=[],
                weights=None,
                parameter_active=True,
                data_type="continuous"
            )
        ],
        opt_info_list=[
            OptimizationInfo(
                name="GA",
                opt_parameters={
                    "num_generations": 10,     # number of iterations of the algorithm
                    "num_points": 10,            # number of points per param to re-create
                    "refinement_x": 0.5,        # in %
                    "refinement_y": 0.5,        # in %
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
        wobbly_example.run(step=i)

    remove_files_after_run()


if __name__ == "__main__":
    main()
