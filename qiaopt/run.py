""" Defines the run"""
import os
import math
import pandas

from qiaopt.pre import Experiment
from qiaopt.optimization import Optimizer, GAOpt, GenericOptimization

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

    Parameters
    ----------
    experiment : Experiment
        The :obj:Experiment that is being run.
    step_number : int
        The number of the current step.
    job_number : int
        The number of the current job.
    """
    source_dir = experiment.system_setup.source_directory
    output_dir = experiment.system_setup.output_directory
    new_working_dir = os.path.join(source_dir, output_dir, f'step{step_number}', f'job{job_number}')

    if not os.path.exists(new_working_dir):
        os.makedirs(new_working_dir)
    experiment.system_setup.working_directory = new_working_dir


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
        print(f"Finished run of {self.experiment.name} (step{step_number}).")

    def submit(self, step_number=0) -> list:
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
        manager = LocalManager()
        stdout = self.experiment.system_setup.stdout_basename

        jobs = Jobs()
        if not self.experiment.data_points:
            raise RuntimeError(f"Can not submit jobs for Experiment {self.experiment.name}: No datapoints available.")

        for i, item in enumerate(self.experiment.data_points):
            set_basic_directory_structure_for_job(experiment=self.experiment, step_number=step_number, job_number=i)
            jobs.add(
                name=self.experiment.name + str(i),
                args=qcgpilot_commandline(self.experiment, datapoint_item=item),
                stdout=stdout + str(i) + ".txt",
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
        Collects data from output files of the current step of the experiment and saves it into a dict that can be
        accessed through `input_params_to_cost_value()`.

        The new goal here should be to construct a lookup table from output.csv (or the output of the scripts)
        which as the keys has tuples of the input parameters and as its value the cost associated with that combination
        of parameters
        aka: output.csv                     ----> self.input_cost_map : dict
        C a b c .... x                          {(1, 5, 3, ..., .1) : .7,
        .7 1 5 3 .. .1                           (...) : ..}

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
            # todo : This is unsorted, is that a problem?
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
        row = self.input_param_cost_df.iloc[solution_idx]
        if all(math.isclose(row[i + 1], solution[i]) for i in range(len(solution))):
            return row[0]
        else:
            raise ValueError(f"Solution {solution} was not found in internal dataframe.")

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

            # todo : this would implement constraints, param type could be subsumed into this maybe by giving 'step'=1?
            # constraints = [param.constraints for param in self.experiment.parameters if param.is_active]
            #
            # param_types = [int if param.param_type == "discrete" else float for param in self.experiment.parameters
            #                if param.is_active]
            # if len(set(param_types)) == 1:
            #     param_types = param_types[0]
            # todo: figure out what's nicer for user constraint or data_type? both seems redundant?
            if opt_info.name == 'GA':
                optimization_alg = GAOpt(initial_population=self.experiment.data_points,
                                         fitness_func=self.input_params_to_cost_value,
                                         # gene_type=param_types,
                                         # gene_space=constraints,
                                         **opt_info.parameters)
            else:
                raise ValueError('Unknown optimization algorithm.')
        else:
            optimization_alg = None

        return optimization_alg


class Executor(Core):
    def __init__(self, experiment: Experiment):
        super().__init__(experiment)

    def run(self, step=0, evolutionary_point_generation=None) -> None:
        super().run(step, evolutionary_point_generation)
