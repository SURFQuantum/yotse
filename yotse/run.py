""" Defines the run"""
import os
import math
from typing import Tuple

import pandas
import pickle

from yotse.pre import Experiment
from yotse.optimization import Optimizer, GAOpt, GenericOptimization

from qcg.pilotjob.api.job import Jobs
from qcg.pilotjob.api.manager import LocalManager


def qcgpilot_commandline(experiment: Experiment, datapoint_item: list) -> list:
    """
     Creates a command line for the QCG-PilotJob executor based on the experiment configuration.

     Parameters:
     -----------
     experiment: Experiment
         The experiment to configure the command line for.
    datapoint_item : list or float #todo : fix this so it always gets a list?
        Datapoint containing the specific values for each parameter e.g. (x1, y2, z1).

     Returns:
     --------
     list
         A list of strings representing the command line arguments for the QCG-PilotJob executor.
     """
    cmdline = [os.path.join(experiment.system_setup.source_directory, experiment.system_setup.program_name)]
    # add parameters
    for p, param in enumerate(experiment.parameters):
        if param.is_active:
            cmdline.append(f"--{param.name}")
            if len(experiment.parameters) == 1:
                # single parameter
                cmdline.append(datapoint_item)
            else:
                cmdline.append(datapoint_item[p])
    # add fixed cmdline arguments
    for key, value in experiment.system_setup.cmdline_arguments.items():
        cmdline.append(key)
        cmdline.append(str(value))
    return cmdline


def get_files_by_extension(directory: str, extension: str) -> list:
    """
    Returns a list of files in the given directory with the specified extension.

    Parameters:
    -----------
    directory: str
        The directory to search for files in.
    extension: str
        The file extension to search for.

    Returns:
    --------
    list
        A list of files (and their actual location) in the given directory with the specified extension.
    """
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(extension)]


def file_list_to_single_df(files: list) -> pandas.DataFrame:
    """
    Reads CSV files from a list and combines their content in a single dataframe.

    Parameters:
    -----------
    files: list
        A list of CSV files to read.

    Returns:
    --------
    df : pandas.Dataframe
        Pandas dataframe containing the combined contents of all the CSV files.
    """
    dfs = [pandas.read_csv(file, delimiter=' ') for file in files]
    return pandas.concat(dfs, ignore_index=True)


class Core:
    """
    Defines the default run function for the executor.

    Parameters:
    -----------
    experiment: Experiment
        The experiment to run.

    Attributes:
    -----------
    optimization_alg : GenericOptimization
        Specific subclass of GenericOptimization that implements the optimization algorithm.
    optimizer : Optimizer
       Object of Optimizer responsible to execute the optimization process.
    num_points : int
        Number of new points for each Parameter to create in each optimization step, specified in the OptimizationInfo.
    refinement_factors : list
        Refinement factors for each of the Parameters specified in the experiment.
    """

    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.optimization_alg = self.set_optimization_algorithm()
        if self.optimization_alg is not None:
            self.optimizer = Optimizer(optimization_algorithm=self.optimization_alg)
        else:
            self.optimizer = None
        self.input_param_cost_df = None

        if "--resume" in self.experiment.system_setup.cmdline_arguments:
            # if resuming the simulation, load state from file
            self.load_core_state(aux_directory=self.experiment.system_setup.cmdline_arguments["--resume"])

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
        _, aux_dir = self.submit(step_number=step_number)
        data = self.collect_data()
        self.create_points_based_on_optimization(data=data, evolutionary=evolutionary_point_generation)
        self.save_core_state(aux_directory=aux_dir)
        print(f"Finished run of {self.experiment.name} (step{step_number}).")

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
        print("aux_dir", aux_dir)


        jobs = Jobs()
        if not self.experiment.data_points:
            raise RuntimeError(f"Can not submit jobs for Experiment {self.experiment.name}: No datapoints available.")

        for i, item in enumerate(self.experiment.data_points):
            cmdline = self.pre_submission_setup_per_job(datapoint_item=item, step_number=step_number, job_number=i)
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
        return job_ids, aux_dir

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
        if self.optimization_alg is not None:
            if evolutionary is None:
                evolutionary = self.optimization_alg.can_create_points_evolutionary

            self.update_internal_cost_data(data=data)

            if self.optimization_alg.function is None:
                raise RuntimeError("Optimization attempted to create new points without a cost function.")
            self.optimizer.optimize()
            self.optimizer.construct_points(experiment=self.experiment,
                                            evolutionary=evolutionary)

    def input_params_to_cost_value(self, ga_instance, solution, solution_idx) -> float:
        """Return value of cost function for given set of input parameter values and their index in the set of points.

        Parameters:
        ----------
        solution : list
            Set of input parameter values of shape [param_1, param_2, .., param_n].
        solution_idx : int
            Index of the solution within the set of points.
        """
        # todo: input parameters of this are highly GA specific and should be made general
        row = self.input_param_cost_df.iloc[solution_idx]
        if all(math.isclose(row[i + 1], solution[i]) for i in range(len(solution))):
            return row[0]
        else:
            raise ValueError(f"Solution {solution} was not found in internal dataframe row {solution_idx}.")

    def update_internal_cost_data(self, data: pandas.DataFrame) -> None:
        """Update internal dataframe mapping input parameters to the associated cost from input data.
        It also checks that the ordering of the entries is the same as the data_points of the experiment.

        Parameters:
        ----------
        data : pandas.Dataframe
            A pandas dataframe containing the collected data in the format cost_value init_param_1 ... init_param_n.
        """
        # check ordering of data versus initial datapoints to avoid mistakes when fetching corresponding cost by index
        if len(data) != len(self.experiment.data_points):
            raise ValueError("Data has a different number of rows than the list of datapoints.")
        for i, values in enumerate(self.experiment.data_points):
            row = data.iloc[i]
            if any(not math.isclose(row[j + 1], values[j]) for j in range(len(values))):
                raise ValueError(f"Position of {values} is different between data and original data_points")

        self.input_param_cost_df = data

    def suggest_best_solution(self) -> list:
        return self.optimization_alg.get_best_solution()

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
        self.set_basic_directory_structure_for_job(self.experiment, step_number, job_number)
        return qcgpilot_commandline(self.experiment, datapoint_item=datapoint_item)

    @staticmethod
    def set_basic_directory_structure_for_job(experiment: Experiment, step_number: int, job_number: int) -> None:
        """
        Creates a new directory for the given step number and updates the experiment's working directory accordingly.

        The basic directory structure is as follows
        source_dir
            - output_dir
                your_run_script.py
                analysis_script.py
                - step_{i}
                     analysis_output.csv
                    - job_{j}
                        output_of_your_run_script.extension
                        stdout{j}.txt

        Parameters:
        ----------
        experiment : Experiment
            The :obj:Experiment that is being run.
        step_number : int
            The number of the current step.
        job_number : int
            The number of the current job.
        """
        source_dir = experiment.system_setup.source_directory
        output_dir = experiment.system_setup.output_dir_name
        new_working_dir = os.path.join(source_dir, output_dir, f'step{step_number}', f'job{job_number}')

        if not os.path.exists(new_working_dir):
            os.makedirs(new_working_dir)
        experiment.system_setup.working_directory = new_working_dir

    def set_optimization_algorithm(self) -> GenericOptimization:
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
            # ref_factors = [param.refinement_factor for param in self.experiment.parameters if param.is_active]
            # if None in ref_factors or len(ref_factors) != len([p for p in self.experiment.parameters if p.is_active]):
            #     raise ValueError("When using refinement factors they must be specified for all active parameters.")

            # Note: param_type could be subsumed into this maybe by giving 'step'=1?
            constraints = [param.constraints for param in self.experiment.parameters if param.is_active]
            # check if there are no constraints
            if all(x is None for x in constraints):
                constraints = None
            # todo: add more tests that check what happens if only some constraints are None etc.
            # param_types = [int if param.param_type == "discrete" else float for param in self.experiment.parameters
            #                if param.is_active]
            # if len(set(param_types)) == 1:
            #     param_types = param_types[0]
            # todo: figure out what's nicer for user constraint or data_type? both seems redundant?
            if opt_info.name == 'GA':
                optimization_alg = GAOpt(initial_population=self.experiment.data_points,
                                         fitness_func=self.input_params_to_cost_value,
                                         # gene_type=param_types,
                                         gene_space=constraints,
                                         **opt_info.parameters)
            else:
                raise ValueError('Unknown optimization algorithm.')
        else:
            optimization_alg = None

        return optimization_alg

    def save_core_state(self, aux_directory):
        """Save state of the core to be able to resume later."""

        with open(os.path.join(aux_directory, 'yotse_state_save.pickle'), 'wb') as file:
            pickle.dump(self.__dict__, file)

    def load_core_state(self, aux_directory):
        """Load the state of the core to be able to resume."""
        try:
            with open(os.path.join(aux_directory, 'yotse_state_save.pickle'), 'rb') as file:
                state = pickle.load(file)
            self.__dict__.update(state)
        except FileNotFoundError:
            raise ValueError(f"No saved state file found in {aux_directory}, when trying to resume workflow.")


class Executor(Core):
    def __init__(self, experiment: Experiment):
        super().__init__(experiment)

    def run(self, step=0, evolutionary_point_generation=None) -> None:
        super().run(step, evolutionary_point_generation)
