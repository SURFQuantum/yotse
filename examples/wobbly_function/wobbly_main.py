"""Example script for execution of a wobbly_function.py experiment."""
import os
from qiaopt.pre import Experiment, SystemSetup, Parameter, OptimizationInfo


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
                number_points=2,
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
                    "order": "cr",
                    "number_best_candidates": "10",
                    "global_scale_factor": "1.0",
                    "population_size": "20",
                    "probability_mutation": "0.5",
                    "probability_crossover": "0.5"
                })]
    )
    return wobbly_experiment
