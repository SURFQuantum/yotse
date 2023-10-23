import os
import pickle
from typing import Tuple

import pandas
from qcg.pilotjob.api.job import Jobs
from qcg.pilotjob.api.manager import LocalManager

from yotse.optimization.algorithms import GAOpt
from yotse.optimization.optimizer import Optimizer
from yotse.pre import Experiment
from yotse.pre import set_basic_directory_structure_for_job
from yotse.utils.utils import file_list_to_single_df
from yotse.utils.utils import get_files_by_extension


class Executor:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.optimizer = self.generate_optimizer()
        self.aux_dir = None

        if "--resume" in self.experiment.system_setup.cmdline_arguments:
            # if resuming the simulation, load state from file
            self.load_executor_state(aux_directory=self.experiment.system_setup.cmdline_arguments["--resume"])

    def generate_optimizer(self) -> Optimizer:
        """Sets the optimization algorithm for the run by translating information in the optimization_info.

        Returns:
        -------
        optimization_alg : GenericOptimization
            Object of subclass of `:class:GenericOptimization`, the optimization algorithm to be used by this runner.
        """
        if self.experiment.optimization_information_list:
            if len([opt for opt in self.experiment.optimization_information_list if opt.is_active]) > 1:
                raise RuntimeError('Multiple active optimization steps. Please set all but one to active=False')
            opt_info = self.experiment.optimization_information_list[0]
            # todo: moving refinement factors to params is also an option
            # ref_factors = [param.refinement_factor for param in experiment.parameters if param.is_active]
            # if None in ref_factors or len(ref_factors) != len([p for p in experiment.parameters if p.is_active]):
            #     raise ValueError("When using refinement factors they must be specified for all active parameters.")

            # Note: param_type could be subsumed into this maybe by giving 'step'=1?
            constraints = [param.constraints for param in self.experiment.parameters if param.is_active]
            # check if there are no constraints
            if all(x is None for x in constraints):
                constraints = None
            # todo: add more tests that check what happens if only some constraints are None etc.
            # param_types = [int if param.param_type == "discrete" else float for param in experiment.parameters
            #                if param.is_active]
            # if len(set(param_types)) == 1:
            #     param_types = param_types[0]
            # todo: figure out what's nicer for user constraint or data_type? both seems redundant?
            if opt_info.name == 'GA':
                optimization_alg = GAOpt(initial_population=self.experiment.data_points,
                                         # gene_type=param_types,
                                         gene_space=constraints,
                                         **opt_info.parameters)
            else:
                raise ValueError('Unknown optimization algorithm.')
        else:
            optimization_alg = None

        return Optimizer(optimization_algorithm=optimization_alg)

    def run(self, step_number=0, evolutionary_point_generation=None) -> None:
        """ Submits jobs to the LocalManager, collects the output, creates new data points, and finishes the run.

        Parameters:
        -----------
        step_number : int (optional)
            Step number to submit to QCGPilot. Should be used for e.g. running different optimization steps.
            Defaults to 0.
        evolutionary_point_generation
            # todo : fill in docstring
        """
        print(f"Starting default run of {self.experiment.name} (step{step_number}): submit, collect, create.")
        self.submit(step_number=step_number)
        data = self.collect_data()
        self.create_points_based_on_optimization(data=data, evolutionary=evolutionary_point_generation)
        self.save_executor_state()
        print(f"Finished run of {self.experiment.name} (step{step_number}).")

    def pre_submission_setup_per_job(self, datapoint_item: list, step_number: int, job_number: int) -> list:
        """Sets up the basic directory structure for a job and returns the QCG-Pilot command line list for it.

        Parameters:
        ----------
        datapoint_item : list
            Single item of data points for the job as a list.
        step_number : int
            The number of the step in the experiment.
        job_number: int
            The number of the job within the step.

        Returns:
        -------
        qcg_cmdline_list : list
            The list of command line arguments for the QCG-Pilot job submission.

        Note: Overwrite this function if you need other directory structure or pre-submission functionality.

        """
        set_basic_directory_structure_for_job(self.experiment, step_number, job_number)
        return self.experiment.qcgpilot_commandline(datapoint_item=datapoint_item)

    def submit(self, step_number=0) -> Tuple[list, str]:
        """
        Submits jobs to the LocalManager.

        Parameters:
        ----------
        step_number : int (optional)
            Step number to submit to QCGPilot. Should be used for e.g. running different optimization steps.
            Defaults to 0.

        Returns:
        --------
        job_ids : list
            A list of job IDs submitted to the LocalManager.
        """
        manager = LocalManager(cfg=self.experiment.system_setup.qcg_cfg)
        stdout = self.experiment.system_setup.stdout_basename
        instance_id = manager.system_status()['System']['InstanceId']
        aux_dir = os.path.join(os.getcwd(), '.qcgpjm-service-{}'.format(instance_id))
        self.aux_dir = aux_dir

        jobs = Jobs()
        if not self.experiment.data_points:
            raise RuntimeError(f"Can not submit jobs for Experiment {self.experiment.name}: No datapoints available.")
        for i, item in enumerate(self.experiment.data_points):
            cmdline = self.pre_submission_setup_per_job(datapoint_item=item,
                                                        step_number=step_number,
                                                        job_number=i)
            jobs.add(
                name=self.experiment.name + str(i),
                args=cmdline,
                stdout=stdout + str(i) + ".txt",
                stderr=stdout + str(i) + ".err",
                wd=self.experiment.system_setup.working_directory,
                **self.experiment.system_setup.job_args
            )
        if self.experiment.system_setup.analysis_script is not None:
            # add analysis job with correct dependency
            jobs.add(
                name=self.experiment.name + f"step{step_number}_analysis",
                args=[os.path.join(self.experiment.system_setup.source_directory,
                                   self.experiment.system_setup.analysis_script)],
                stdout=stdout + f"step{step_number}_analysis.txt",
                stderr=stdout + f"step{step_number}_analysis.err",
                wd=self.experiment.system_setup.current_step_directory,
                after=jobs.job_names(),
                **self.experiment.system_setup.job_args
            )
        job_ids = manager.submit(jobs)
        manager.wait4(job_ids)
        manager.finish()
        manager.cleanup()
        return job_ids

    def collect_data(self) -> pandas.DataFrame:
        """
        Collects data from output.csv (or the output of the scripts) and combines it into a dataframe which has as
        first column the associated cost and as the other columns the input parameters (order the same way is input to
        the experiment).
        The rows of the dataframe follow the same ordering as the jobs.

        Returns:
        -------
        data : pandas.Dataframe
            Pandas dataframe containing the combined outputs of the individual jobs in the form above.

        """
        if self.experiment.system_setup.analysis_script is None:
            # no analysis script: extract data from output files in job dirs and combine to single dataframe
            output_directory_current_step = self.experiment.system_setup.current_step_directory
            extension = self.experiment.system_setup.output_extension
            files = []
            for job_dir in [x[0] for x in os.walk(output_directory_current_step)
                            if x[0] != output_directory_current_step]:
                files.extend(get_files_by_extension(job_dir, extension))
            data = file_list_to_single_df(files)
            # todo : This is unsorted, is that a problem? yes. sort this by job no.
        else:
            # analysis script is given and will output file 'output.csv' with format 'cost_fun param0 param1 ...'
            data = pandas.read_csv(os.path.join(self.experiment.system_setup.current_step_directory, 'output.csv'),
                                   delim_whitespace=True)
        return data

    def create_points_based_on_optimization(self, data: pandas.DataFrame, evolutionary=None) -> None:
        """
        Applies an optimization algorithm to process the collected data and create new data points from it which is then
        directly written into the experiments attributes.

        Parameters:
        ----------
        data : pandas.Dataframe
            A pandas dataframe containing the collected data in the format cost_value init_param_1 ... init_param_n.
        evolutionary : bool (optional)
            Overwrite the type of construction to be used for the new points. If evolutionary=None the optimization
            algorithm determines whether the point creation is evolutionary or based on the best solution.
            Defaults to None.
        """
        if self.optimizer.optimization_algorithm is not None:
            if evolutionary is None:
                evolutionary = self.optimizer.optimization_algorithm.can_create_points_evolutionary

            self.optimizer.optimization_algorithm.update_internal_cost_data(experiment=self.experiment, data=data)

            if self.optimizer.optimization_algorithm.function is None:
                raise RuntimeError("Optimization attempted to create new points without a cost function.")
            self.optimizer.optimize()
            self.optimizer.construct_points(experiment=self.experiment, evolutionary=evolutionary)

    def save_executor_state(self):
        """Save state of the Executor to be able to resume later."""
        # Note: maybe this should be optional, because not everything might be serializable, e.g. complex cost_functions
        with open(os.path.join(self.aux_dir, 'yotse_state_save.pickle'), 'wb') as file:
            pickle.dump(self.__dict__, file)
        print(f"Latest state of yotse executor saved in {self.aux_dir}.")

    def load_executor_state(self, aux_directory):
        """Load the state of the Executor to be able to resume."""
        try:
            with open(os.path.join(aux_directory, 'yotse_state_save.pickle'), 'rb') as file:
                state = pickle.load(file)
            self.__dict__.update(state)
            print(f"State of yotse executor loaded from {self.aux_dir}.")
        except FileNotFoundError:
            raise ValueError(f"No saved state file found in {aux_directory}, when trying to resume workflow.")


class CustomExecutor(Executor):

    def __init__(self, experiment: Experiment):
        super().__init__(experiment)

    def run(self, step_number=0, evolutionary_point_generation=None):
        super().run(step_number=step_number, evolutionary_point_generation=evolutionary_point_generation)
