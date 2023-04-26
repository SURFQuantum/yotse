import os
import shutil
import matplotlib

from qiaopt.pre import Experiment, SystemSetup, Parameter, OptimizationInfo
from qiaopt.run import Core
from qiaopt.blueprint_tools import setup_optimization_dir, create_separate_files_for_job


def blueprint_input():
    blueprint_experiment = Experiment(
        experiment_name="DelftEindhovenNVSURFDoubleClick",
        system_setup=SystemSetup(
            # Note : here it is important to write the absolute path, since we
            source_directory="/home/davidm/Projects/QiaOpt/examples/blueprint_example/src",
            program_name="unified_simulation_script_state_with_translation.py",
            command_line_arguments={"configfile": "nv_surf_config.yaml",
                                    "paramfile": "nv_baseline_params.yaml",
                                    "--n_runs": 100},
            analysis_script="processing_function.py",
            executor="/home/davidm/Projects/QiaOpt/examples/blueprint_example/src/venv_wrapper.sh",
            # files_needed=("*.py", "*.pickle", "*.yaml"),  # todo: not implemented yet
            output_directory="output",
            # output_extension=".csv",  # collect_data would pick these up if there is no analysis script
            venv="~/Projects/venvs/qcg-venv"  # todo venv not working? no packages installed?
            # todo: missing queue, time_run, time_analysis
        ),
        parameters=[
            Parameter(
                name="detector_efficiency",
                param_range=[0.9, 0.99999],
                constraints={'low': 0.9, 'high': 0.99999},
                number_points=3,
                distribution="uniform",
                param_type="continuous",
                # scale_factor=0.1
            ),
            Parameter(
                name="n1e",
                param_range=[5300, 50000],
                constraints={'low': 5300, 'high': 50000},
                number_points=2,
                distribution="uniform",
                param_type="continuous",
                # scale_factor=44700
            ),
            Parameter(
                name="visibility",
                param_range=[0.9, 0.99999],
                constraints={'low': 0.9, 'high': 0.99999},
                number_points=2,
                distribution="uniform",
                param_type="continuous",
                # scale_factor=0.1
            ),
            Parameter(
                name="ec_gate_depolar_prob",
                param_range=[0.0001, 0.02],
                constraints={'low': 0.0001, 'high': 0.02},
                number_points=2,
                distribution="uniform",
                param_type="continuous",
                # scale_factor=0.02
            ),
            Parameter(
                name="carbon_T2",
                param_range=[1e+9, 1e+10],
                constraints={'low': 1e+9, 'high': 1e+10},
                number_points=2,
                distribution="uniform",
                param_type="continuous",
                # scale_factor=1e+10
            ),
            Parameter(
                name="electron_T2",
                param_range=[5e+8, 1e+10],
                constraints={'low': 5e+8, 'high': 1e+10},
                number_points=2,
                distribution="uniform",
                param_type="continuous",
                # scale_factor=1e+10
            ),
            Parameter(
                name="cutoff_time",
                param_range=[0.01, 1.],
                constraints={'low': 0.01, 'high': 1.},
                number_points=2,
                distribution="uniform",
                param_type="continuous",
                # scale_factor=0.99,
                depends_on={'name': "carbon_T2",
                            'function': linear_dep}
            )
        ],
        opt_info_list=[
            OptimizationInfo(
                name="GA",
                opt_parameters={
                    # "num_generations": 200,
                    "num_generations": 50,
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


def linear_dep(x, y):
    return x*y


def remove_files_after_run():
    # remove files and directories
    shutil.rmtree('../output')
    dirs = [f for f in os.listdir(os.getcwd()) if (f.startswith(".qcg"))]
    for d in dirs:
        shutil.rmtree(os.path.join(os.getcwd(), d))

    os.remove('venv_wrapper.sh')


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
        # workaround to enable proper virtualenv usage
        executor_name = os.path.basename(experiment.system_setup.job_args["exec"])
        self.create_venv_wrapper(venv_path=experiment.system_setup.job_args["venv"], wrapper_filename=executor_name)

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


def main():
    def cost_function(f):
        return f

    experiment = blueprint_input()
    experiment.cost_function = cost_function
    blueprint_example = BlueprintExecutor(experiment=experiment)

    for i in range(experiment.optimization_information_list[0].parameters["num_generations"]):
        blueprint_example.run(step=i)

    solution = blueprint_example.suggest_best_solution()
    print("Solution: ", solution)
    matplotlib.use('Qt5Agg')
    # wobbly_example.optimization_alg.ga_instance.plot_new_solution_rate()
    blueprint_example.optimization_alg.ga_instance.plot_fitness()
    remove_files_after_run()


if __name__ == "__main__":
    main()
