"""This module demonstrates the process for a blackbox bayesian optimization using the
Yotse framework."""
from bayes_opt import UtilityFunction

from examples.wobbly_function.example_wobbly_main import wobbly_pre
from yotse.execution import Executor
from yotse.pre import OptimizationInfo


def main() -> None:
    """Executes a blackbox optimization experiment for a 'wobbly function' of unknown
    form (at least we pretend to not know for the purpose of this example)..

    This function first adds the bayesian optimization information, sets it to active
    and finally runs the optimization..
    """
    print("\033[93m --- Executing Wobbly-Bayesian Example. --- \033[0m")

    bayesopt_experiment = wobbly_pre()

    bayes_opt = OptimizationInfo(
        name="bayesopt",
        blackbox_optimization=True,
        opt_parameters={
            "utility_function": UtilityFunction(kind="ucb", kappa=2.5, xi=0.0),
            "n_iter": 10,
        },
        is_active=True,
    )
    # deactivate other optimization
    bayesopt_experiment.opt_info_list[0].is_active = False
    # append new active whitebox optimization
    bayesopt_experiment.opt_info_list.append(bayes_opt)

    wobbly_bayesopt = Executor(experiment=bayesopt_experiment)

    for i in range(bayes_opt.opt_parameters["n_iter"]):
        wobbly_bayesopt.run(step_number=i)

    solution = wobbly_bayesopt.optimizer.suggest_best_solution()
    print("Solution: ", solution)
    # matplotlib.use('Qt5Agg')
    # wobbly_example.optimization_alg.ga_instance.plot_new_solution_rate()
    # wobbly_example.optimization_alg.ga_instance.plot_fitness()
    # remove_files_after_run()


if __name__ == "__main__":
    main()
