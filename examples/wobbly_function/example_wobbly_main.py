"""Example script for execution of a wobbly_function.py experiment."""
import os
from qiaopt.pre import Experiment, SystemSetup, Parameter, OptimizationInfo
from qiaopt.run import Core, set_basic_directory_structure_for_job, qcgpilot_commandline
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


class WobblyCore(Core):

    def create_points_based_on_method(self, data):
        def wobbly_cost_fun(x, y):
            return x**2 - y**2

        ga_opt = GAOpt(function=wobbly_cost_fun, data=data, num_generations=100)
        optimizer = Optimizer(ga_opt)
        solution, func_values = optimizer.optimize()
        xy_new, func_new = optimizer.construct_points(solution,
                                                      num_points=5,
                                                      delta_x=0.5,
                                                      delta_y=0.5)
        # TODO: missing is the possibility to pass parameters to the GAOpt
        return xy_new

    def submit(self, step_number=0):
        """
        Submits jobs to the LocalManager.

        Returns:
        --------
        list
            A list of job IDs submitted to the LocalManager.
        """
        manager = LocalManager()
        stdout = self.experiment.system_setup.stdout_basename

        jobs = Jobs()
        for i, item in enumerate(self.experiment.data_points):
            set_basic_directory_structure_for_job(experiment=self.experiment, step_number=step_number, job_number=i)
            jobs.add(
                name=self.experiment.name + str(i),
                exec=self.experiment.system_setup.executor,
                args=qcgpilot_commandline(self.experiment),
                stdout=stdout + str(i) + ".txt",
                wd=self.experiment.system_setup.working_directory,
            )
            # analysis_job
            # Todo: do we want to standardize this call as well?
            # eg. in the parent function include if analysis_script is not None: add_analysis_job()
            jobs.add(
                name=self.experiment.name + str(i) + "analysis",
                exec=self.experiment.system_setup.executor,
                args=[os.path.join(self.experiment.system_setup.source_directory,
                                   self.experiment.system_setup.analysis_script)],
                stdout=stdout + str(i) + "analysis.txt",
                wd=self.experiment.system_setup.working_directory,
            )
        job_ids = manager.submit(jobs)
        manager.wait4(job_ids)
        manager.finish()
        manager.cleanup()
        return job_ids

    def run(self):
        for step in range(10):
            self.submit(step)
            data = self.collect_data()
            self.create_points_based_on_method(data)


def main():
    wobbly_example = WobblyCore(experiment=wobbly_pre())
    wobbly_example.run()


if __name__ == "__main__":
    main()
    # add clean up function that removes qcg folders (see test-run.py)
