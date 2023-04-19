import os
import shutil

from qiaopt.pre import Experiment, SystemSetup, Parameter, OptimizationInfo
from qiaopt.run import Core
from qiaopt.blueprint_tools import setup_optimization_dir, create_separate_files_for_job


def blueprint_input():
    blueprint_experiment = Experiment(
        experiment_name="DelftEindhovenNVSURFDoubleClick",
        system_setup=SystemSetup(
            source_directory=os.getcwd(),
            program_name="unified_simulation_script_state_with_translation.py",
            command_line_arguments={"configfile": "nv_surf_config.yaml",
                                    "paramfile": "nv_baseline_params.yaml",
                                    "--n_runs": 100},
            analysis_script="processing_function.py",
            executor="/home/davidm/Projects/QiaOpt/examples/blueprint_example/src/venv_wrapper.sh",
            files_needed=("*.py", "*.pickle", "*.yaml"),
            output_directory="output",
            # output_extension=".csv",  # collect_data would pick these up if there is no analysis script
            venv="~/Projects/venvs/qcg-venv"  # todo venv not working? no packages installed?
            # todo: missing queue, time_run, time_analysis
        ),
        parameters=[
            Parameter(
                name="detector_efficiency",
                param_range=[0.9, 0.99999],
                number_points=3,
                distribution="uniform",
                param_type="continuous",
                # scale_factor=0.1
            ),
            Parameter(
                name="n1e",
                param_range=[5300, 50000],
                number_points=2,
                distribution="uniform",
                param_type="continuous",
                # scale_factor=44700
            ),
            Parameter(
                name="visibility",
                param_range=[0.9, 0.99999],
                number_points=2,
                distribution="uniform",
                param_type="continuous",
                # scale_factor=0.1
            ),
            Parameter(
                name="ec_gate_depolar_prob",
                param_range=[0.0001, 0.02],
                number_points=2,
                distribution="uniform",
                param_type="continuous",
                # scale_factor=0.02
            ),
            Parameter(
                name="carbon_T2",
                param_range=[1e+9, 1e+10],
                number_points=2,
                distribution="uniform",
                param_type="continuous",
                # scale_factor=1e+10
            ),
            Parameter(
                name="electron_T2",
                param_range=[5e+8, 1e+10],
                number_points=2,
                distribution="uniform",
                param_type="continuous",
                # scale_factor=1e+10
            ),
            Parameter(
                name="cutoff_time",
                param_range=[0.01, 1.],
                number_points=2,
                distribution="uniform",
                param_type="continuous",
                # scale_factor=0.99
            )
        ],
        opt_info_list=[
            OptimizationInfo(
                name="GA",
                opt_parameters={
                    "num_generations": 200,
                    # "maximum": False,
                    "num_parents_mating": 20,           # todo was missing in blueprint code
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
                is_active=True
            )
        ]
    )

    return blueprint_experiment


def remove_files_after_run():
    # remove files and directories
    shutil.rmtree('output')
    dirs = [f for f in os.listdir(os.getcwd()) if (f.startswith(".qcg"))]
    for d in dirs:
        shutil.rmtree(os.path.join(os.getcwd(), d))


class BlueprintCore(Core):
    """Executor implementation using adaptions for NLBlueprint."""
    def pre_submission_setup_per_job(self, datapoint_item: list, step_number: int, job_number: int) -> None:
        setup_optimization_dir(experiment=self.experiment, step_number=step_number, job_number=job_number)
        new_cmdline = create_separate_files_for_job(experiment=self.experiment, datapoint_item=datapoint_item,
                                                    step_number=step_number, job_number=job_number)

        return new_cmdline


class BlueprintExecutor(BlueprintCore):
    def __init__(self, experiment: Experiment):
        super().__init__(experiment)

    def run(self, step=0, evolutionary_point_generation=None) -> None:
        super().run(step, evolutionary_point_generation)


def main():
    def cost_function(f):
        return f

    experiment = blueprint_input()
    experiment.cost_function = cost_function
    blueprint_example = BlueprintExecutor(experiment=experiment)

    for i in range(experiment.optimization_information_list[0].parameters["num_generations"]):
        blueprint_example.run(step=i)

    remove_files_after_run()


if __name__ == "__main__":
    main()
