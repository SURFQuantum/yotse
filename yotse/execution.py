"""Defines classes and functions for the execution of your experiment."""
import os
import pickle
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas
from qcg.pilotjob.api.job import Jobs
from qcg.pilotjob.api.manager import LocalManager

from yotse.optimization.blackbox_algorithms import BayesOpt
from yotse.optimization.blackbox_algorithms import GAOpt
from yotse.optimization.generic_optimization import GenericOptimization
from yotse.optimization.optimizer import Optimizer
from yotse.optimization.whitebox_algorithms import SciPyOpt
from yotse.pre import Experiment
from yotse.pre import OptimizationInfo
from yotse.pre import set_basic_directory_structure_for_job
from yotse.utils.utils import file_list_to_single_df
from yotse.utils.utils import get_files_by_extension


class Executor:
    """A facilitator for running experiments and optimization algorithms.

    The `Executor` class coordinates the execution of experiments, manages the optimization process,
    and interfaces with the LocalManager for job submission. It supports both black-box and white-box
    optimization strategies, allowing for the exploration of various optimization algorithms.

    Attributes
    ----------
    experiment : Experiment
        The experiment object associated with the executor.
    blackbox_optimization : bool
        Flag indicating whether the optimization process is black-box or white-box.
    optimizer : Optimizer
        The optimizer object responsible for managing the optimization algorithm.
    aux_dir : str
        The auxiliary directory used during the optimization process.

    Parameters
    ----------
    experiment : Experiment
        The experiment object associated with the executor.
    """

    def __init__(self, experiment: Experiment):
        """Initialize `Executor` object."""
        self.experiment: Experiment = experiment
        self.blackbox_optimization = True
        self.optimizer: Optimizer = self.generate_optimizer()
        self.aux_dir: str = ""

        if "--resume" in self.experiment.system_setup.cmdline_arguments:
            assert isinstance(
                self.experiment.system_setup.cmdline_arguments["--resume"], str
            ), "--resume keyword must be passed a string describing the path to the aux directory."
            # if resuming the simulation, load state from file
            self.load_executor_state(
                aux_directory=self.experiment.system_setup.cmdline_arguments["--resume"]
            )

    def next_optimization(self) -> None:
        """Switch to the next active optimization in the list.

        Deactivates the current active optimization and activates the next one in the list.

        Raises:
            RuntimeError: If there are multiple active optimizations or none at all.
        """
        active_optimizations: List[OptimizationInfo] = [
            opt for opt in self.experiment.opt_info_list if opt.is_active
        ]

        if len(active_optimizations) == 1:
            active_optimization = active_optimizations[0]
            active_index = self.experiment.opt_info_list.index(active_optimization)
            next_index = (active_index + 1) % len(self.experiment.opt_info_list)

            # Deactivate the current optimization
            active_optimization.is_active = False

            # Activate the next optimization
            self.experiment.opt_info_list[next_index].is_active = True
        elif len(active_optimizations) > 1:
            raise RuntimeError(
                "Multiple active optimization steps. Please set all but one to active=False"
            )
        else:
            raise RuntimeError("No active optimization steps found.")

        self.optimizer = self.generate_optimizer()

    def get_active_optimization(self) -> OptimizationInfo:
        """Get the active optimization step.

        Returns
        -------
        OptimizationInfo
            The active optimization step.

        Raises
        ------
        RuntimeError
            If there are multiple active optimization steps.
        RuntimeError
            If no active optimization steps are found.
        """

        active_optimizations = [
            opt for opt in self.experiment.opt_info_list if opt.is_active
        ]

        if len(active_optimizations) == 1:
            return active_optimizations[0]
        elif len(active_optimizations) > 1:
            raise RuntimeError(
                "Multiple active optimization steps. Please set all but one to active=False"
            )
        else:
            raise RuntimeError("No active optimization steps found.")

    def generate_optimizer(self) -> Optimizer:
        """Sets the optimization algorithm for the run by translating information in the
        currently 'active' optimization_info.

        Returns
        -------
        optimization_alg : GenericOptimization
            Object of subclass of `:class:GenericOptimization`, the optimization algorithm to be used by this runner.
        """
        optimization_alg: Optional[GenericOptimization] = None

        if self.experiment.opt_info_list:
            opt_info = self.get_active_optimization()

            if opt_info.blackbox_optimization:
                self.blackbox_optimization = True
                # todo: moving refinement factors to params is also an option
                # ref_factors = [param.refinement_factor for param in experiment.parameters if param.is_active]
                # if None in ref_factors or len(ref_factors) != len([p for p in experiment.parameters if p.is_active]):
                #     raise ValueError("When using refinement factors they must be specified for all active parameters.")

                # Note: param_type could be subsumed into this maybe by giving 'step'=1?
                constraints = [
                    param.constraints
                    for param in self.experiment.parameters
                    if param.is_active
                ]
                # check if there are no constraints
                if all(x is None for x in constraints):
                    constraints = None  # type: ignore
                # todo: add more tests that check what happens if only some constraints are None etc.
                # param_types = [int if param.param_type == "discrete" else float for param in experiment.parameters
                #                if param.is_active]
                # if len(set(param_types)) == 1:
                #     param_types = param_types[0]
                # todo: figure out what's nicer for user constraint or data_type? both seems redundant?
                if opt_info.name.lower() == "ga":
                    optimization_alg = GAOpt(
                        blackbox_optimization=True,
                        initial_data_points=self.experiment.data_points,
                        # gene_type=param_types,
                        gene_space=constraints,  # type: ignore
                        **opt_info.opt_parameters,
                    )
                elif opt_info.name.lower() == "test":
                    optimization_alg = None
                elif opt_info.name.lower() == "bayesopt":
                    optimization_alg = BayesOpt(
                        blackbox_optimization=True,
                        initial_data_points=self.experiment.data_points,
                        pbounds={
                            param.name: (
                                int(param.range[0]),
                                int(param.range[1]),
                            )  # cast to int for bayesian opt, see comment in class
                            for param in self.experiment.parameters
                        },
                        **opt_info.opt_parameters,
                    )
                else:
                    raise ValueError(
                        f"Unknown blackbox optimization algorithm: {opt_info.name}"
                    )
            else:
                # whitebox optimization
                self.blackbox_optimization = False

                # if opt_info.name.lower() == "ga":
                #     optimization_alg = GAOpt(
                #         blackbox_optimization=False,
                #         fitness_func=opt_info.function,
                #         initial_data_points=self.experiment.data_points,
                #         # gene_type=param_types,
                #         gene_space=constraints,  # type: ignore
                #         **opt_info.opt_parameters,
                #     )
                if opt_info.name.lower() == "scipy":
                    optimization_alg = SciPyOpt(**opt_info.opt_parameters)
                elif opt_info.name.lower() == "test":
                    optimization_alg = None
                else:
                    raise ValueError(
                        f"Unknown whitebox optimization algorithm: {opt_info.name}"
                    )

        return Optimizer(optimization_algorithm=optimization_alg)  # type: ignore[arg-type]

    def run(
        self, step_number: int = 0, evolutionary_point_generation: Optional[bool] = None
    ) -> None:
        """Submits jobs to the LocalManager, collects the output, creates new data
        points, and finishes the run.

        Parameters
        ----------
        step_number : int (optional)
            Step number to submit to QCGPilot. Should be used for e.g. running different optimization steps.
            Defaults to 0.
        evolutionary_point_generation : bool (optional)
            Overwrite the type of construction to be used for the new points. If None the optimization
            algorithm determines whether the point creation is evolutionary or based on the best solution.
            Defaults to None.
        """
        if self.blackbox_optimization:
            print(
                f"Starting default run of {self.experiment.name} (step{step_number}): submit, collect, create."
            )
            self.submit(step_number=step_number)
            data = self.collect_data()
            self.create_points_based_on_optimization(
                data=data, evolutionary=evolutionary_point_generation
            )
            self.save_executor_state()
            print(f"Finished run of {self.experiment.name} (step{step_number}).")
        else:
            print(
                f"Starting run of {self.experiment.name} using whitebox optimization {self.get_active_optimization().name}."
            )
            self.whitebox_submit()

    def pre_submission_setup_per_job(
        self, datapoint_item: List[float], step_number: int, job_number: int
    ) -> List[Union[str, Any]]:
        """Sets up the basic directory structure for a job and returns the QCG- Pilot
        command line list for it.

        Parameters
        ----------
        datapoint_item : list
            Single item of data points for the job as a list.
        step_number : int
            The number of the step in the experiment.
        job_number : int
            The number of the job within the step.

        Returns
        -------
        program_commandline : list
            The list of command line arguments for the QCG-Pilot job submission for the program.
        Note: Overwrite this function if you need other directory structure or pre-submission functionality.
        """
        assert not isinstance(
            datapoint_item, np.ndarray
        )  # check item is not an array, which is not serializable
        set_basic_directory_structure_for_job(self.experiment, step_number, job_number)
        program_commandline = self.experiment.qcgpilot_commandline(
            datapoint_item=datapoint_item
        )
        return program_commandline

    def pre_submission_analysis(self) -> List[Union[str, Any]]:
        """Executes any necessary steps before the analysis script and returns the QCG-
        Pilot command line list for it.

        Returns
        -------
        analysis_commandline : list
            The list of command line arguments for the QCG-Pilot job submission for the program.
        Note: Overwrite this function if you need other directory structure or pre-submission functionality for your
        analysis script.
        """
        assert self.experiment.system_setup.analysis_script is not None
        analysis_commandline = [
            os.path.join(
                self.experiment.system_setup.source_directory,
                self.experiment.system_setup.analysis_script,
            )
        ]
        return analysis_commandline

    def submit(self, step_number: int = 0) -> List[str]:
        """Submits jobs to the LocalManager.

        Parameters
        ----------
        step_number : int, optional
            Step number to submit to QCGPilot. Should be used for e.g. running different optimization steps.
            Defaults to 0.

        Returns
        -------
        job_ids : list
            A list of job IDs submitted to the LocalManager.
        """

        manager = LocalManager(cfg=self.experiment.system_setup.qcg_cfg)
        stdout = self.experiment.system_setup.stdout_basename
        instance_id = manager.system_status()["System"]["InstanceId"]
        aux_dir = os.path.join(os.getcwd(), ".qcgpjm-service-{}".format(instance_id))
        self.aux_dir = aux_dir

        jobs = Jobs()
        if self.experiment.data_points.size == 0:
            raise RuntimeError(
                f"Can not submit jobs for Experiment {self.experiment.name}: No datapoints available."
            )
        for i, item in enumerate(self.experiment.data_points):
            prog_cmdline = self.pre_submission_setup_per_job(
                datapoint_item=item.tolist(), step_number=step_number, job_number=i
            )
            jobs.add(
                {
                    "name": self.experiment.name + str(i),
                    "args": prog_cmdline,
                    "stdout": stdout + str(i) + ".txt",
                    "stderr": stdout + str(i) + ".err",
                    "wd": self.experiment.system_setup.working_directory,
                },
                **self.experiment.system_setup.job_args,
            )
        if self.experiment.system_setup.analysis_script is not None:
            analysis_cmdline = self.pre_submission_analysis()
            # add analysis job with correct dependency
            jobs.add(
                {
                    "name": self.experiment.name + f"step{step_number}_analysis",
                    "args": analysis_cmdline,
                    "stdout": stdout + f"step{step_number}_analysis.txt",
                    "stderr": stdout + f"step{step_number}_analysis.err",
                    "wd": self.experiment.system_setup.current_step_directory,
                    "after": jobs.job_names(),
                },
                **self.experiment.system_setup.job_args,
            )
        job_ids = manager.submit(jobs)
        manager.wait4(job_ids)
        manager.finish()
        manager.cleanup()
        return job_ids  # type: ignore[no-any-return]

    def whitebox_submit(self) -> None:
        """Run the white-box optimization process.

        Currently, this does not use QCGPilotJob but runs locally.
        """
        # todo: change this to run using QCQPilotJob
        self.optimizer.optimize()
        return

    def collect_data(self) -> pandas.DataFrame:
        """Collects data from output.csv (or the output of the scripts) and combines it
        into a dataframe which has as first column the associated cost and as the other
        columns the input parameters (order the same way is input to the experiment).
        The rows of the dataframe follow the same ordering as the jobs.

        Returns
        -------
        data : pandas.Dataframe
            Pandas dataframe containing the combined outputs of the individual jobs in the form above.
        """
        if self.experiment.system_setup.analysis_script is None:
            # no analysis script: extract data from output files in job dirs and combine to single dataframe
            output_directory_current_step = (
                self.experiment.system_setup.current_step_directory
            )
            extension = self.experiment.system_setup.output_extension
            files = []
            for job_dir in [
                x[0]
                for x in os.walk(output_directory_current_step)
                if x[0] != output_directory_current_step
            ]:
                files.extend(get_files_by_extension(job_dir, extension))
            data = file_list_to_single_df(files, extension)
            # todo : This is unsorted, is that a problem? yes. sort this by job no.
        else:
            # analysis script is given and will output file 'output.csv' with format 'cost_fun param0 param1 ...'
            data = pandas.read_csv(
                os.path.join(
                    self.experiment.system_setup.current_step_directory, "output.csv"
                ),
                delim_whitespace=True,
            )
        return data

    def create_points_based_on_optimization(
        self, data: pandas.DataFrame, evolutionary: Optional[bool] = None
    ) -> None:
        """Applies an optimization algorithm to process the collected data and create
        new data points from it which is then directly written into the experiments
        attributes.

        Parameters
        ----------
        data : pandas.Dataframe
            A pandas dataframe containing the collected data in the format cost_value init_param_1 ... init_param_n.
        evolutionary : bool , optional
            Overwrite the type of construction to be used for the new points. If evolutionary=None the optimization
            algorithm determines whether the point creation is evolutionary or based on the best solution.
            Defaults to None.
        """
        if self.optimizer.optimization_algorithm is not None:
            if evolutionary is None:
                evolutionary = (
                    self.optimizer.optimization_algorithm.can_create_points_evolutionary
                )

            self.optimizer.update_blackbox_cost_data(
                experiment=self.experiment, data=data
            )

            if self.optimizer.optimization_algorithm.function is None:
                raise RuntimeError(
                    "Optimization attempted to create new points without a cost function."
                )
            self.optimizer.optimize()
            self.optimizer.construct_points(
                experiment=self.experiment, evolutionary=evolutionary
            )

    def save_executor_state(self) -> None:
        """Save state of the Executor to be able to resume later."""
        # Note: maybe this should be optional, because not everything might be serializable, e.g. complex cost_functions
        with open(os.path.join(self.aux_dir, "yotse_state_save.pickle"), "wb") as file:
            pickle.dump(self.__dict__, file)
        print(f"Latest state of yotse executor saved in {self.aux_dir}.")

    def load_executor_state(self, aux_directory: str) -> None:
        """Load the state of the Executor to be able to resume."""
        try:
            with open(
                os.path.join(aux_directory, "yotse_state_save.pickle"), "rb"
            ) as file:
                state = pickle.load(file)
            self.__dict__.update(state)
            print(f"State of yotse executor loaded from {self.aux_dir}.")
        except FileNotFoundError:
            raise ValueError(
                f"No saved state file found in {aux_directory}, when trying to resume workflow."
            )


class CustomExecutor(Executor):
    """Custom Executor class for users to tailor to their specific experimental setups.

    The `CustomExecutor` class is a user-defined extension of the base `Executor` class.
    Users can customize this class to adapt the optimization and execution process to
    their specific experimental requirements.

    Parameters
    ----------
    experiment : Experiment
        The experiment object associated with the custom executor.
    """

    def __init__(self, experiment: Experiment):
        """Initialize `CustomExecutor` object."""
        super().__init__(experiment)

    def run(
        self, step_number: int = 0, evolutionary_point_generation: Optional[bool] = None
    ) -> None:
        """Run the custom execution process.

        This method overrides the run method in the base `Executor` class to provide
        custom logic for the execution process tailored to the user's specific needs.

        Parameters
        ----------
        step_number : int, optional
            Step number to submit to QCGPilot. Should be used for e.g. running different optimization steps.
            Defaults to 0.
        evolutionary_point_generation : bool, optional
            Overwrite the type of construction to be used for the new points.
            If None, the optimization algorithm determines whether the point creation is evolutionary or
            based on the best solution. Defaults to None.
        """
        super().run(
            step_number=step_number,
            evolutionary_point_generation=evolutionary_point_generation,
        )
