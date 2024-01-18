"""This module demonstrates the process for parameter switching during optimization
using the Yotse framework."""
from examples.wobbly_function.example_wobbly_main import remove_files_after_run
from examples.wobbly_function.example_wobbly_main import wobbly_pre
from yotse.execution import Executor
from yotse.pre import Parameter


def main() -> None:
    """Executes a two blackbox optimization experiment for a 'wobbly function' of
    unknown form.

    This function first adds another parameter (inactive) then GA optimizes over params
    1 and 2. Then it follows this up with another GA optimization over parameters 2 and
    3.
    """
    print("\033[93m --- Executing Wobbly-Param-Switching Example. --- \033[0m")

    param_switch_experiment = wobbly_pre()
    # add third inactive parameter
    param_switch_experiment.add_parameter(
        Parameter(
            name="z",
            param_range=[1, 4],
            number_points=10,
            distribution="linear",
            parameter_active=False,
        )
    )

    param_switch_example = Executor(experiment=param_switch_experiment)

    for i in range(
        param_switch_experiment.opt_info_list[0].opt_parameters["num_generations"]
    ):
        param_switch_example.run(step_number=i, evolutionary_point_generation=True)

    solution = param_switch_example.optimizer.suggest_best_solution()
    print("Solution after first GA: ", solution)

    # Switching active params from param 1 & 2 to param 2 & 3
    param_switch_example.experiment.parameters[0].parameter_active = False
    param_switch_example.experiment.parameters[2].parameter_active = True
    # create new datapoints with new active params
    param_switch_example.experiment.create_datapoint_c_product()
    # todo: maybe nice additional functionality could be that this is done automatically when params have changed.

    for i in range(
        param_switch_experiment.opt_info_list[0].opt_parameters["num_generations"],
        2 * param_switch_experiment.opt_info_list[0].opt_parameters["num_generations"],
    ):
        param_switch_example.run(step_number=i, evolutionary_point_generation=True)

    solution = param_switch_example.optimizer.suggest_best_solution()
    print("Solution after second GA: ", solution)

    remove_files_after_run()


if __name__ == "__main__":
    main()
