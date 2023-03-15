"""Example script for execution of a wobbly_function.py experiment."""
import os
from qiaopt.pre import Experiment, SystemSetup, Parameter, OptimizationInfo
from qiaopt.run import Executor
from qiaopt.optimization import Optimizer, GAOpt


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
                    "num_generations": 100,     # number of iterations of the algorithm
                    "num_points": 5,            # number of points to re-create
                    "refinement_x": 0.5,        # in %
                    "refinement_y": 0.5,        # in %
                    "logging_level": 1,
                },
                is_active=True)]
    )
    return wobbly_experiment


class WobblyExecutor(Executor):

    def __init__(self, experiment):
        super().__init__(experiment)
        opt_info = self.experiment.optimization_information_list[0]
        self.refinement_x = opt_info.parameters['refinement_x']
        self.refinement_y = opt_info.parameters['refinement_y']
        self.num_points = opt_info.parameters['num_points']
        if opt_info.name == 'GA':
            num_generation = opt_info.parameters['num_generations']
            self.optimization_alg = GAOpt(function=self.cost_function,
                                          data=None,
                                          num_generations=opt_info.parameters['num_generations'],
                                          logging_level=opt_info.parameters['logging_level'])
        else:
            print('Error! Unknown optimization algorithm.')
            exit(1)

        self.optimizer = Optimizer(self.optimization_alg)

    @staticmethod
    def cost_function(x, y):
        return x**2 - y**2

    def create_points_based_on_method(self, data):
        # ga_opt = GAOpt(function=self.cost_function, data=data, num_generations=100)
        # optimizer = Optimizer(ga_opt)
        self.optimization_alg.data = data
        print("data", data)
        solution, func_values = self.optimizer.optimize()

        xy_new, func_new = self.optimizer.construct_points(solution,
                                                           num_points=self.num_points,
                                                           refinement_x=self.refinement_x,
                                                           refinement_y=self.refinement_y)
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
