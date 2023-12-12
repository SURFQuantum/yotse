"""This module provides a blueprint for setting up and executing an optimization problem
within the NlBlueprint setup of the Quantum Internet Alliance project using a genetic
algorithm (GA) and the Yotse execution framework.

It defines the experiment setup, including the system configuration, optimization
parameters, and the GA optimization information. The module also includes a custom
executor class tailored for the NlBlueprint team and utility functions for pre-
submission setup and post-run cleanup.
"""
import os
import shutil
from typing import Any
from typing import List
from typing import Union

import matplotlib

from yotse.execution import Executor
from yotse.pre import Experiment
from yotse.pre import OptimizationInfo
from yotse.pre import Parameter
from yotse.pre import SystemSetup
from yotse.utils.blueprint_tools import create_separate_files_for_job
from yotse.utils.blueprint_tools import setup_optimization_dir


def blueprint_input():
    """Constructs and returns an Experiment instance configured for the NlBlueprint use-
    case.

    This includes the system setup, parameter configuration for the experiment, and optimization
    information specific to a genetic algorithm. Some parameters have been commented out to simplify
    the example; they can be uncommented and adjusted for actual optimization scenarios.

    Returns
    -------
    Experiment
        The configured Experiment object with system setup, parameter specification,
        and genetic algorithm optimization information.
    """
    blueprint_experiment = Experiment(
        experiment_name="DelftEindhovenNVSURFDoubleClick",
        system_setup=SystemSetup(
            # Note : here it is important to write the absolute path, since we
            source_directory=os.getcwd(),
            program_name="unified_simulation_script_state_with_translation.py",
            command_line_arguments={
                "configfile": "nv_surf_config.yaml",
                "paramfile": "nv_baseline_params.yaml",
                "--n_runs": 100,
            },
            analysis_script="processing_function.py",
            executor="python",
            output_dir_name="output",
            venv=os.environ.get(
                "BLUEPRINT_VENV_PATH", "/home/runner/work/yotse/yotse/blueprint_venv"
            ),  # the environment where you installed all dependencies for your script
            slurm_venv="~/yotse_venv",  # the environment where you installed yotse
            num_nodes=2,
            alloc_time="01:00:00",
            slurm_args=["--exclusive"],
            qcg_cfg={"log_level": "DEBUG"},
            modules=[
                "2022",
                "Python/3.10.4-GCCcore-11.3.0",
            ],  # These are not needed if you set a slurm_venv
        ),
        parameters=[
            Parameter(
                name="detector_efficiency",
                param_range=[0.9, 0.99999],
                constraints={"low": 0.9, "high": 0.99999},
                number_points=3,
                distribution="uniform",
                param_type="continuous",
            ),
            Parameter(
                name="n1e",
                param_range=[5300, 50000],
                constraints={"low": 5300, "high": 50000},
                number_points=2,
                distribution="uniform",
                param_type="continuous",
            ),
            # Note: Commenting out some params to improve runtime of this example. If using this for actual optimization
            # please uncomment parameters and increase the number of generations.
            #
            # Parameter(
            #     name="visibility",
            #     param_range=[0.9, 0.99999],
            #     constraints={"low": 0.9, "high": 0.99999},
            #     number_points=2,
            #     distribution="uniform",
            #     param_type="continuous",
            # ),
            # Parameter(
            #     name="ec_gate_depolar_prob",
            #     param_range=[0.0001, 0.02],
            #     constraints={"low": 0.0001, "high": 0.02},
            #     number_points=2,
            #     distribution="uniform",
            #     param_type="continuous",
            # ),
            Parameter(
                name="carbon_T2",
                param_range=[1e9, 1e10],
                constraints={"low": 1e9, "high": 1e10},
                number_points=2,
                distribution="uniform",
                param_type="continuous",
            ),
            Parameter(
                name="electron_T2",
                param_range=[5e8, 1e10],
                constraints={"low": 5e8, "high": 1e10},
                number_points=2,
                distribution="uniform",
                param_type="continuous",
            ),
            Parameter(
                name="cutoff_time",
                param_range=[0.01, 1.0],
                constraints={"low": 0.01, "high": 1.0},
                number_points=2,
                distribution="uniform",
                param_type="continuous",
                depends_on={"name": "carbon_T2", "function": linear_dep},
                # todo: test if this dependency is also used in each generation
            ),
        ],
        opt_info_list=[
            OptimizationInfo(
                name="GA",
                blackbox_optimization=True,
                opt_parameters={
                    # "num_generations": 200,
                    "num_generations": 2,
                    # "maximum": False,
                    "num_parents_mating": 20,  # todo was missing in blueprint code
                    # "global_scale_factor": 1.0,       what is this supposed to do?
                    # "number_parameters": 7,           Unnecessary, num params not determined internally
                    # "number_best_candidates": 40,
                    "keep_elitism": 40,
                    # "population_size": 200,           Obsolete, population size determined by initial population
                    # "proba_mutation": 0.02,
                    "mutation_probability": 0.02,
                    # "proba_crossover": 0.7,
                    "crossover_probability": 0.7,
                },
                is_active=True,
            )
        ],
    )

    return blueprint_experiment


def linear_dep(x, y):
    """A sample linear dependency function that returns the product of two input values.

    Parameters
    ----------
    x : float
        The first input value.
    y : float
        The second input value.

    Returns
    -------
    float
        The product of `x` and `y`.
    """
    return x * y


def remove_files_after_run():
    """Cleans up by removing output directories and QCG-related directories created
    during the run."""
    # remove files and directories
    shutil.rmtree("../output")
    dirs = [f for f in os.listdir(os.getcwd()) if (f.startswith(".qcg"))]
    for d in dirs:
        shutil.rmtree(os.path.join(os.getcwd(), d))


class BlueprintCore(Executor):
    """An Executor subclass tailored for the NlBlueprint, with adaptations for specific
    setup requirements and analysis preparation before job submission.

    Methods
    -------
    pre_submission_setup_per_job(datapoint_item, step_number, job_number)
        Configures optimization directories and creates separate job files.
    pre_submission_analysis()
        Prepares analysis command line arguments before job submission.
    """

    def pre_submission_setup_per_job(
        self, datapoint_item: List[float], step_number: int, job_number: int
    ) -> List[Union[str, Any]]:
        """Sets up optimization directories and creates separate files for each job as
        part of pre-submission setup.

        Parameters
        ----------
        datapoint_item : list of float
            The data point for the job.
        step_number : int
            The step number in the optimization process.
        job_number : int
            The job number for the current step.

        Returns
        -------
        list of str or any
            The modified command line arguments for the job.
        """
        setup_optimization_dir(
            experiment=self.experiment, step_number=step_number, job_number=job_number
        )
        new_cmdline = create_separate_files_for_job(
            experiment=self.experiment,
            datapoint_item=datapoint_item,
            step_number=step_number,
            job_number=job_number,
        )

        return new_cmdline

    def pre_submission_analysis(self) -> List[Union[str, Any]]:
        """Prepares the analysis command line before submission, ensuring the analysis
        script and necessary arguments are included.

        Returns
        -------
        list of str or any
            The command line arguments for the analysis script.
        """
        assert self.experiment.system_setup.analysis_script is not None
        analysis_commandline = [
            os.path.join(
                self.experiment.system_setup.source_directory,
                self.experiment.system_setup.analysis_script,
            )
        ]
        analysis_commandline.append("--paramfile")
        analysis_commandline.append(
            self.experiment.system_setup.cmdline_arguments["paramfile"]
        )
        analysis_commandline.append("--variedparams")
        analysis_commandline.extend(
            [param.name for param in self.experiment.parameters if param.is_active]
        )
        return analysis_commandline


class BlueprintExecutor(BlueprintCore):
    """The primary executor class for running the blueprint example.

    Inherits from BlueprintCore and provides a run method to execute the experiment.
    """

    def __init__(self, experiment: Experiment):
        """Initializes the BlueprintExecutor with the given experiment setup.

        Parameters
        ----------
        experiment : Experiment
            The experiment configuration for execution.
        """
        super().__init__(experiment)

    def run(self, step=0, evolutionary_point_generation=None) -> None:
        """Runs the optimization for a given step, delegating to the parent class's run
        method.

        Parameters
        ----------
        step : int, optional
            The current optimization step number, by default 0.
        evolutionary_point_generation : optional
            Method of generating new points based on evolutionary principles, by default None.
        """
        super().run(step, evolutionary_point_generation)


def main(plot=False):
    """The main execution function for the blueprint example. Sets up the experiment,
    runs the optimization process, and handles the output and cleanup.

    Parameters
    ----------
    plot : bool, optional
        Whether to plot the fitness graph at the end of the execution, by default False.
    """
    print("\033[93m --- Executing NlBlueprint Example --- \033[0m")

    experiment = blueprint_input()
    blueprint_example = BlueprintExecutor(experiment=experiment)
    # Note: For the blueprint example we are not defining a cost function in yotse, as the cost funtion is defined in
    # the module `processing_function.py` and is based on parameters yotse never extracts. Yotse only takes the
    # associated cost returned by `processing_function.py` and minimizes that.

    experiment.parse_slurm_arg("example_blueprint_main.py")

    for i in range(experiment.opt_info_list[0].opt_parameters["num_generations"]):
        blueprint_example.run(step=i)

    # output
    # todo what do we want to output in the end? should this file also create a stdout
    solution = blueprint_example.optimizer.suggest_best_solution()
    print("Solution: ", solution)
    with open("solution.txt", "w") as file:
        file.write(f"Solution: {solution} \n")
    if plot:
        # plot fitness
        matplotlib.use("Qt5Agg")
        # wobbly_example.optimization_alg.ga_instance.plot_new_solution_rate()
        (
            fig,
            ax,
        ) = blueprint_example.optimizer.optimization_algorithm.optimization_instance.plot_fitness()
        fig.savefig("fitness.png")

    # clean up
    remove_files_after_run()


if __name__ == "__main__":
    main()
