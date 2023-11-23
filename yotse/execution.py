"""Defines classes and functions for the execution of your experiment."""
import logging
import os
import pickle
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas
from qcg.pilotjob.api.job import Jobs
from qcg.pilotjob.api.manager import LocalManager

from yotse.optimization.algorithms import GAOpt
from yotse.optimization.optimizer import Optimizer
from yotse.pre import Experiment
from yotse.pre import set_basic_directory_structure_for_job
from yotse.utils.utils import file_list_to_single_df
from yotse.utils.utils import get_files_by_extension

logger = logging.getLogger(__name__)


class Executor:
    def __init__(self, experiment: Experiment):
        self.experiment: Experiment = experiment
        self.optimizer: Optimizer = self.generate_optimizer()
        self.aux_dir: str = ""

        if "--resume" in self.experiment.system_setup.cmdline_arguments:
            resume_arg = self.experiment.system_setup.cmdline_arguments["--resume"]
            if not isinstance(resume_arg, str):
                logger.error(
                    "--resume keyword must be passed a string describing the path to the aux directory."
                )
                raise ValueError(
                    "--resume keyword must be passed a string describing the path to the aux directory."
                )
            else:
                # if resuming the simulation, load state from file
                self.load_executor_state(aux_directory=resume_arg)

    def _set_job_manager(self) -> LocalManager:
        """Set the job manager that will be used for this Executor."""
        # Note: if this is set on init, a process is started and never finishes, so only set up just before submit
        # Note: this is rather annoying as this means a new LocalManager needs to be created every iteration
        logger.info("Setting up LocalManager.")
        manager = LocalManager(
            ["--log", "error"], cfg=self.experiment.system_setup.qcg_cfg
        )
        logger.info(f"QCQ-PiloJob reports available resources: {manager.resources()}")
        return manager

    @staticmethod
    def extract_aux_dir(job_manager: LocalManager) -> str:
        """Helper function to extract the auxiliary directory where qcg-pilotjob files
        and saves are stored."""
        logger.info("Getting aux_dir.")
        instance_id = job_manager.system_status()["System"]["InstanceId"]
        aux_dir = os.path.join(os.getcwd(), ".qcgpjm-service-{}".format(instance_id))
        return aux_dir

    def generate_optimizer(self) -> Optimizer:
        """Sets the optimization algorithm for the run by translating information in the
        optimization_info.

        Returns
        -------
        optimization_alg : GenericOptimization
            Object of subclass of `:class:GenericOptimization`, the optimization algorithm to be used by this runner.
        """
        if self.experiment.optimization_information_list:
            if (
                len(
                    [
                        opt
                        for opt in self.experiment.optimization_information_list
                        if opt.is_active
                    ]
                )
                > 1
            ):
                raise RuntimeError(
                    "Multiple active optimization steps. Please set all but one to active=False"
                )
            opt_info = self.experiment.optimization_information_list[0]
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
            if opt_info.name == "GA":
                optimization_alg = GAOpt(
                    initial_data_points=self.experiment.data_points,
                    # gene_type=param_types,
                    gene_space=constraints,  # type: ignore
                    **opt_info.parameters,
                )
            else:
                raise ValueError("Unknown optimization algorithm.")
        else:
            optimization_alg = None

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
        logger.info(
            f"Starting default run of {self.experiment.name} (step{step_number}): submit, collect, create."
        )
        self.submit(step_number=step_number)
        data = self.collect_data()
        self.create_points_based_on_optimization(
            data=data, evolutionary=evolutionary_point_generation
        )
        self.save_executor_state()
        logger.info(f"Finished run of {self.experiment.name} (step{step_number}).")

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
        stdout = self.experiment.system_setup.stdout_basename

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

        # set up LocalManager and submit
        job_manager = self._set_job_manager()
        self.aux_dir = self.extract_aux_dir(job_manager=job_manager)

        job_ids = self._submit(manager=job_manager, jobs=jobs)
        job_manager.finish()
        job_manager.cleanup()
        return job_ids

    def _submit(self, manager: LocalManager, jobs: Jobs) -> List[str]:
        """Private function submitting the jobs."""
        logging.info("Submitting jobs.")
        job_ids = manager.submit(jobs)
        logging.info(f"QCQPilotJob submitted jobs: {str(job_ids)}")

        manager.wait4(job_ids)

        assert self._check_job_status(job_status=manager.status(job_ids))
        logging.debug(f"DEBUG: job_info: {manager.info(job_ids)}")
        return job_ids  # type: ignore[no-any-return]

    @staticmethod
    def _check_job_status(job_status: Dict[str, Any]) -> bool:
        """Check the status of the jobs and raise RuntimeError immediately if any
        fail."""
        logging.info(f"Job status after wait: {job_status}")
        if not all(
            job["data"]["status"] == "SUCCEED" for job in job_status["jobs"].values()
        ):
            failed_job = next(
                job
                for job in job_status["jobs"].values()
                if job["data"]["status"] != "SUCCEED"
            )
            raise RuntimeError(f"Job {failed_job['id']} did not succeed")
        logging.info("All Jobs succeeded.")
        return True

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

            self.optimizer.optimization_algorithm.update_internal_cost_data(
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
    def __init__(self, experiment: Experiment):
        super().__init__(experiment)

    def run(
        self, step_number: int = 0, evolutionary_point_generation: Optional[bool] = None
    ) -> None:
        super().run(
            step_number=step_number,
            evolutionary_point_generation=evolutionary_point_generation,
        )
