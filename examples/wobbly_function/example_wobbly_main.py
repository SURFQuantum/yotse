"""Example script for execution of a wobbly_function.py experiment."""
import os
from qiaopt.pre import Experiment, SystemSetup, Parameter, OptimizationInfo
from qiaopt.run import Executor, set_basic_directory_structure_for_job, qcgpilot_commandline
from qiaopt.optimization import Optimizer, GAOpt
from qcg.pilotjob.api.manager import LocalManager
from qcg.pilotjob.api.job import Jobs


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


class WobblyExecutor(Executor):

    @staticmethod
    def cost_function(x, y):
        return x**2 - y**2

    def create_points_based_on_method(self, data):
        ga_opt = GAOpt(function=self.cost_function, data=data, num_generations=100)
        optimizer = Optimizer(ga_opt)
        solution, func_values = optimizer.optimize()
        xy_new, func_new = optimizer.construct_points(solution,
                                                      num_points=5,
                                                      delta_x=0.5,
                                                      delta_y=0.5)
        # TODO: missing is the possibility to pass parameters to the GAOpt
        return xy_new


def main():
    num_opt_steps = 10
    wobbly_example = WobblyExecutor(experiment=wobbly_pre())
    for i in range(num_opt_steps):
        wobbly_example.run(step=i)


if __name__ == "__main__":
    main()
    # add clean up function that removes qcg folders (see test-run.py)
