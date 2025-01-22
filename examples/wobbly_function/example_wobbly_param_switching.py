"""This module demonstrates the process for parameter switching during optimization
using the Yotse framework."""

from examples.wobbly_function.example_wobbly_main import remove_files_after_run
from examples.wobbly_function.example_wobbly_main import wobbly_pre
from yotse.execution import Executor
from yotse.pre import ConstraintDict
from yotse.pre import OptimizationInfo
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
            param_range=[4, 5],
            constraints=ConstraintDict(low=4, high=10),
            number_points=10,
            distribution="linear",
            parameter_active=False,
        )
    )
    second_ga_opt = OptimizationInfo(
        name="GA",
        blackbox_optimization=True,
        opt_parameters={
            "num_generations": 4,  # number of iterations of the algorithm
            # "num_points": 10,            # number of points per param to re-create , now determined by initial
            "num_parents_mating": 5,
            "mutation_probability": 0.2,
            "refinement_factors": [0.5, 0.5],
            "logging_level": 1,
        },
        is_active=False,
    )
    del param_switch_experiment.opt_info_list[
        1
    ]  # delete whitebox optimization that is still in the list
    param_switch_experiment.opt_info_list.append(second_ga_opt)
    num_gens_1 = param_switch_experiment.opt_info_list[0].opt_parameters[
        "num_generations"
    ]
    num_gens_2 = param_switch_experiment.opt_info_list[1].opt_parameters[
        "num_generations"
    ]

    param_switch_example = Executor(experiment=param_switch_experiment)

    for i in range(num_gens_1):
        param_switch_example.run(step_number=i, evolutionary_point_generation=True)

    solution = param_switch_example.optimizer.suggest_best_solution()
    print("Solution after first GA: ", solution)

    # Switching active params from param 1 & 2 to param 2 & 3
    param_switch_example.experiment.parameters[0].parameter_active = False
    param_switch_example.experiment.parameters[2].parameter_active = True
    # create new datapoints with new active params and write them to experiment
    param_switch_example.experiment.data_points = (
        param_switch_example.experiment.create_datapoint_c_product()
    )
    # todo: maybe nice additional functionality could be that this is done automatically when params have changed.

    param_switch_example.next_optimization()  # switching to next opt which is GA again

    for i in range(num_gens_1, num_gens_1 + num_gens_2):
        param_switch_example.run(step_number=i, evolutionary_point_generation=True)

    solution = param_switch_example.optimizer.suggest_best_solution()
    print("Solution after second GA: ", solution)

    remove_files_after_run()


if __name__ == "__main__":
    main()
