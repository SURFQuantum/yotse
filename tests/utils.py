"""Utility functions for use in unittests."""

import os
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from yotse.execution import Executor
from yotse.pre import ConstraintDict
from yotse.pre import Experiment
from yotse.pre import OptimizationInfo
from yotse.pre import Parameter
from yotse.pre import ParameterDependencyDict
from yotse.pre import SystemSetup


if os.getcwd().endswith("tests"):
    DUMMY_FILE = "myfunction.py"
else:
    DUMMY_FILE = "tests/myfunction.py"


def create_default_param(
    name: str = "bright_state_parameter",
    parameter_range: List[Union[float, int]] = [0.1, 0.9],
    number_points: int = 9,
    distribution: str = "linear",
    constraints: Union[ConstraintDict, np.ndarray, None] = None,
    custom_distribution: Optional[Callable[[float, float, int], np.ndarray]] = None,
    param_type: str = "continuous",
    parameter_active: bool = True,
    depends_on: Optional[ParameterDependencyDict] = None,
) -> Parameter:
    """Return a default Parameter instance with optional custom settings."""
    return Parameter(
        name=name,
        param_range=parameter_range,
        number_points=number_points,
        distribution=distribution,
        constraints=constraints,
        custom_distribution=custom_distribution,
        param_type=param_type,
        parameter_active=parameter_active,
        depends_on=depends_on,
    )


def create_default_experiment(
    parameters: Optional[List[Parameter]] = None,
    opt_info_list: Optional[List[OptimizationInfo]] = None,
) -> Experiment:
    """Return a default Experiment instance with optional parameters and optimization
    info."""
    return Experiment(
        experiment_name="default_exp",
        system_setup=SystemSetup(
            source_directory=os.getcwd(),
            program_name=DUMMY_FILE,
            command_line_arguments={"arg1": 0.1, "arg2": "value2"},
        ),
        parameters=parameters,
        opt_info_list=opt_info_list,
    )


def create_default_executor(experiment: Experiment) -> Executor:
    """Instantiate and return an Executor with the given experiment."""
    return Executor(experiment=experiment)
