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
            # executor="python",
            # Note: Test a workaround for the venv bug on snellius
            executor="~/yotse/examples/blueprint_example/src/venv_wrapper.sh",
            output_dir_name="output",
            venv=os.environ.get(
                "BLUEPRINT_VENV_PATH", "/home/runner/work/yotse/yotse/blueprint_venv"
            ),
            slurm_venv="~/yotse_venv",
            num_nodes=2,
            alloc_time="01:00:00",
            slurm_args=["--exclusive"],
            qcg_cfg={"log_level": "DEBUG"},
            modules=["2021", "Python/3.9.5-GCCcore-10.3.0"],
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
                depends_on={"name": "carbon_T2", "function": linear_dep}
                # todo: test if this dependency is also used in each generation
            ),
        ],
        opt_info_list=[
            OptimizationInfo(
                name="GA",
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
    return x * y


def remove_files_after_run():
    # remove files and directories
    shutil.rmtree("../output")
    dirs = [f for f in os.listdir(os.getcwd()) if (f.startswith(".qcg"))]
    for d in dirs:
        shutil.rmtree(os.path.join(os.getcwd(), d))


class BlueprintCore(Executor):
    """Executor implementation using adaptions for NLBlueprint."""

    def pre_submission_setup_per_job(
        self, datapoint_item: List[float], step_number: int, job_number: int
    ) -> List[Union[str, Any]]:
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
    def __init__(self, experiment: Experiment):
        super().__init__(experiment)
        # workaround to enable proper virtualenv usage
        executor_name = os.path.basename(experiment.system_setup.job_args["exec"])
        if executor_name.endswith(".sh"):
            self.create_venv_wrapper(
                venv_path=experiment.system_setup.job_args["venv"],
                wrapper_filename=executor_name,
            )
            # remove venv from job_arg to take over from qcgpilotjob
            del self.experiment.system_setup.job_args["venv"]

    @staticmethod
    def create_venv_wrapper(venv_path: str, wrapper_filename: str = "venv_wrapper.sh"):
        """Workaround function to make QCGPilot use the proper executor from the virtual environment."""
        content = f"""#!/bin/bash
# Activate the virtual environment
source {venv_path}/bin/activate
# Run the main Python script with arguments
python "$@"
"""

        with open(wrapper_filename, "w") as wrapper_file:
            wrapper_file.write(content)

        # Make the wrapper file executable
        import os

        os.chmod(wrapper_filename, os.stat(wrapper_filename).st_mode | 0o111)

    def run(self, step=0, evolutionary_point_generation=None) -> None:
        super().run(step, evolutionary_point_generation)


def cost_function(f):
    """Important note: Cost function should not be defined within main."""
    return f


def main(plot=False):
    experiment = blueprint_input()
    experiment.cost_function = cost_function
    blueprint_example = BlueprintExecutor(experiment=experiment)

    experiment.parse_slurm_arg("example_blueprint_main.py")

    for i in range(
        experiment.optimization_information_list[0].parameters["num_generations"]
    ):
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
        ) = (
            blueprint_example.optimizer.optimization_algorithm.optimization_instance.plot_fitness()
        )
        fig.savefig("fitness.png")

    # clean up
    remove_files_after_run()


if __name__ == "__main__":
    main()
