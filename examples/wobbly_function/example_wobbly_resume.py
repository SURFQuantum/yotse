"""This module demonstrates the process to stop an ongoing experiment and resume it
using the Yotse framework.

It initializes the optimization process, runs for a few generations, and then showcases
how to resume the process from where it left off.
"""
from examples.wobbly_function.example_wobbly_main import remove_files_after_run
from examples.wobbly_function.example_wobbly_main import wobbly_pre
from yotse.execution import Executor


def main() -> None:
    """Executes the stopping and resuming of an optimization experiment for a 'wobbly
    function'.

    This function first runs a part of the optimization process, simulates a stop, and
    then resumes the optimization from the last saved state, finally cleaning up any
    residual files after completion.
    """
    print("\033[93m --- Executing Wobbly-Stop->Resume Example. --- \033[0m")

    # Note: Here, we show how a stopped experiment can be resumed from a save file
    stop_resume_experiment = wobbly_pre()
    stop_resume_example = Executor(experiment=stop_resume_experiment)
    for i in range(3):
        stop_resume_example.run(step_number=i, evolutionary_point_generation=True)

    # write resume parameter to experiment
    resume_experiment = wobbly_pre()
    resume_experiment.system_setup.cmdline_arguments[
        "--resume"
    ] = stop_resume_example.aux_dir

    resume_example = Executor(experiment=resume_experiment)
    for i in range(
        resume_example.optimizer.optimization_algorithm.optimization_instance.generations_completed,
        resume_experiment.optimization_information_list[0].parameters[
            "num_generations"
        ],
    ):
        assert (
            resume_example.optimizer.optimization_algorithm.optimization_instance.generations_completed
            == i
        )
        resume_example.run(step_number=i, evolutionary_point_generation=True)

    resumed_solution = resume_example.optimizer.suggest_best_solution()

    print("Resumed solution: ", resumed_solution)

    remove_files_after_run()


if __name__ == "__main__":
    main()
