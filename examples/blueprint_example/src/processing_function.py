"""This script reads the parameters used in the optimization procedure from the input
file, processes the simulation data generated with them, computes the cost according to
the cost function defined in total_cost and writes it, together with the optimization
parameters to a csv file in a format that can be read by smart-stopos and use to
generate new sets of parameters."""

import os
import pickle
from argparse import ArgumentParser
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
import yaml
from netsquid_netconf.netconf import Loader
from netsquid_nv.nv_parameter_set import _gaussian_dephasing_fn
from netsquid_nv.nv_parameter_set import compute_dephasing_prob_from_nodephasing_number
from netsquid_simulationtools.repchain_data_process import process_data_duration
from netsquid_simulationtools.repchain_data_process import (
    process_data_teleportation_fidelity,
)
from netsquid_simulationtools.repchain_data_process import (
    process_repchain_dataframe_holder,
)

PLATFORM_TO_COHERENCE_TIME = {
    "nv": "carbon_T2",
    "ti": "coherence_time",
    "abstract": "T2",
}

TO_PROB_NO_ERROR_FUNCTION = {
    "detector_efficiency": lambda x: x,
    "collection_efficiency": lambda x: x,
    "p_double_exc": lambda x: 1 - x,
    "ec_gate_depolar_prob": lambda x: 1 - x,
    "n1e": lambda x: 1 - compute_dephasing_prob_from_nodephasing_number(x),
    "electron_T1": lambda x: np.exp(-1 / x),
    "electron_T2": lambda x: np.exp(-1 / x),
    "carbon_T1": lambda x: np.exp(-1 / x),
    "carbon_T2": lambda x: np.exp(-1 / x),
    "std_electron_electron_phase_drift": lambda x: _gaussian_dephasing_fn(x),
    "T1": lambda x: np.exp(-1 / x),
    "T2": lambda x: np.exp(-1 / x),
    "coherence_time": lambda x: np.exp(-1 * (1 / x) ** 2),
    "emission_fidelity": lambda x: x,
    "swap_quality": lambda x: x,
    "visibility": lambda x: x,
    "dark_count_probability": lambda x: 1 - x,
}


def parameter_cost(row: pd.Series, baseline_parameters: dict) -> float:
    """Computes the cost of parameters in `row` with respect to the baseline_parameters.

    Parameters
    ----------
    row : pd.Series
        A pandas Series object containing the values of parameters for which the cost will be computed.
    baseline_parameters : dict
        A dictionary where the keys are the names of optimized hardware parameters and the values are their baseline
         values.

    Returns
    -------
    float
        The hardware parameter cost.
    """
    parameter_cost = 0
    baseline_prob_no_error_dict = {}
    prob_no_error_dict = {}
    for parameter, value in baseline_parameters.items():
        # Some TI parameters are passed as improvement factors, hence they can just directly be added to the cost
        if "improvement" in parameter:
            continue
        baseline_prob_no_error_dict[parameter] = TO_PROB_NO_ERROR_FUNCTION[parameter](
            value
        )
        prob_no_error_dict[parameter] = TO_PROB_NO_ERROR_FUNCTION[parameter](
            row[parameter]
        )
    for parameter in baseline_parameters:
        if "improvement" in parameter:
            parameter_cost += row[parameter]
        else:
            parameter_cost += 1 / (
                np.log(prob_no_error_dict[parameter])
                / np.log(baseline_prob_no_error_dict[parameter])
            )
    return parameter_cost


def total_cost_squared_difference(
    row: pd.Series,
    fidelity_threshold: float,
    rate_threshold: float,
    baseline_parameters: dict,
) -> float:
    """Computes total cost, which includes hardware parameter cost and penalties for not
    meeting target metrics.

    A square difference penalty is used, ensuring that the penalty is higher the furthest away from the target a
    parameter set's performance was.

    Parameters
    ----------
    row : :class:`pandas.DataFrame` row
        Contains values of parameters for which cost will be computed
    fidelity_threshold : float
        Average teleportation fidelity target.
    rate_threshold : float
        Entanglement generation rate target.
    baseline_parameters : dict
        Dictionary where the keys are names of optimized hardware parameters and the values are their baseline values.

    Returns
    -------
    total_cost : float
        Total cost, including hardware parameter cost and penalties.
    """
    fid_cost = (
        1 + (fidelity_threshold - row["teleportation_fidelity_average"]) ** 2
    ) * np.heaviside(fidelity_threshold - row["teleportation_fidelity_average"], 0)
    rate_cost = (1 + (rate_threshold - row["rate"]) ** 2) * np.heaviside(
        rate_threshold - row["rate"], 0
    )
    total_cost = 1.0e20 * (fid_cost + rate_cost) + parameter_cost(
        row, baseline_parameters
    )
    return total_cost


# def parse_from_input_file(filename: str = "input_file.ini") -> list:
#     """Gets list of parameters that were optimized over from input file.
#
#     Parameters
#     ----------
#     filename : str, optional
#         Name of the input file used in the optimization. Defaults to "input_file.ini".
#
#     Returns
#     -------
#     parameter_names : list
#         List of names of the parameters that were optimized over.
#
#     """
#     parameter_names = []
#     parameters = False
#     with open(filename, "r") as f:
#         lines = [line.strip() for line in f.readlines()]
#     for line in lines:
#         variable = line.split(":")[0].strip()
#         if variable == 'Parameter':
#             parameters = True
#         if parameters:
#             if variable == 'name':
#                 name = str(line.split(":")[1].strip())
#                 parameter_names.append(name)
#         if parameters and line.rstrip() == 'end':
#             parameters = False
#
#     return parameter_names


def get_baseline_parameters(
    baseline_parameter_file: str, parameter_list: List[str]
) -> Dict[str, float]:
    """Identifies baseline values of the parameters in `parameter_list`, i.e. the
    parameters that were optimized over. Removes tunable parameters, as those are not
    relevant for computing the cost.

    Parameters
    ----------
    baseline_parameter_file : str
        Name of baseline parameter file.
    parameter_list : list
        List with names of optimized parameters.

    Returns
    -------
    baseline_parameters : dict
        Dictionary where the keys are names of optimized hardware parameters and the values are their baseline values.
    """
    tunable_parameters = [
        "cutoff_time",
        "bright_state_param",
        "coincidence_time_window",
    ]
    with open(baseline_parameter_file, "r") as stream:
        sim_params = yaml.load(stream, Loader=Loader)
    baseline_parameters = {}
    for parameter in parameter_list:
        baseline_parameters[parameter] = sim_params[parameter]
    for parameter in tunable_parameters:
        try:
            del baseline_parameters[parameter]
        except KeyError:
            continue

    return baseline_parameters


def process_data(parameter_list: list, root_dir: str = ".") -> pd.DataFrame:
    """Process raw simulation data keeping the same order as the jobs were submitted in.

    Parameters
    ----------
    parameter_list : list
        List with names of optimized parameters.
    root_dir : str
        Path to the root directory where subdirectories with simulation data are located.
        Defaults to the current directory.

    Returns
    -------
    processed_data : :class:`pandas.DataFrame`
        Dataframe holding processed data.
    """
    processed_data = pd.DataFrame()

    job_dirs = [
        d
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("job")
    ]
    job_dirs.sort(key=lambda x: int(x[3:]))  # Sort job directories by job number

    for job_dir in job_dirs:
        print(job_dir)
        current_job_dir = os.path.join(root_dir, job_dir)
        raw_data_subfolder = "raw_data"  # Replace this with the correct subfolder name
        subfolder_path = os.path.join(current_job_dir, raw_data_subfolder)

        for _, _, subfolder_files in os.walk(subfolder_path):
            for filename in subfolder_files:
                if filename.endswith(".pickle"):
                    file_path = os.path.join(subfolder_path, filename)
                    print(
                        f"Loading pickle file: {file_path}"
                    )  # Add this line to print the file path
                    new_data = pickle.load(open(file_path, "rb"))
                    # manually write varied param to dataframe
                    for param in parameter_list:
                        new_data.copy_baseline_parameter_to_column(name=param)
                    # overwrite varied parameters
                    new_data._reset_varied_parameters()
                    new_data._varied_parameters = parameter_list

                    new_processed_data = process_repchain_dataframe_holder(
                        repchain_dataframe_holder=new_data,
                        processing_functions=[
                            process_data_duration,
                            process_data_teleportation_fidelity,
                        ],
                    )
                    processed_data = processed_data.append(new_processed_data)
    return processed_data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-pf", "--paramfile", required=True, type=str, help="Name of the parameter file"
    )
    parser.add_argument(
        "-vp",
        "--variedparams",
        nargs="+",
        required=True,
        type=str,
        help="Names of the varied parameters",
    )
    args, unknown = parser.parse_known_args()
    # Replace this by the name of the baseline parameter file you used
    # This name should be in the format "platform_baseline_params.yaml"
    baseline_parameter_file = args.paramfile
    # baseline_parameter_file = "nv_baseline_params.yaml"

    param_list = args.variedparams
    # param_list = parse_from_input_file()
    platform = baseline_parameter_file.split("_baseline")[0]
    print("here", baseline_parameter_file, param_list, type(param_list))
    baseline_parameters = get_baseline_parameters(baseline_parameter_file, param_list)
    fid_threshold = 0.8717
    rate_threshold = 0.1

    processed_data = process_data(param_list)
    # sort data by first scan_param # todo : find out why, because for our code we want to preserve job order
    # processed_data.sort_values(by=param_list[0], inplace=True)
    # save processed data
    processed_data.to_csv("full_output.csv", index=False)

    # get output data ready for stopos
    # this means getting the cost in the first column, and the values of the optimized parameters in the other columns

    csv_output = pd.read_csv("full_output.csv")
    output_for_stopos = pd.DataFrame(
        csv_output["teleportation_fidelity_average"],
        columns=["teleportation_fidelity_average"],
    )
    output_for_stopos["rate"] = 1 / csv_output["duration_per_success"]

    # Assuming cutoff is being optimized as factor of coherence time
    try:
        csv_output["cutoff_time"] = (
            csv_output["cutoff_time"] / csv_output[PLATFORM_TO_COHERENCE_TIME[platform]]
        )
    except KeyError:
        pass
    for parameter in param_list:
        output_for_stopos[parameter] = csv_output[parameter]
    output_for_stopos.insert(
        0,
        "cost",
        output_for_stopos.apply(
            lambda row: total_cost_squared_difference(
                row, fid_threshold, rate_threshold, baseline_parameters
            ),
            axis=1,
        ),
    )

    output_for_stopos.drop("teleportation_fidelity_average", inplace=True, axis=1)
    output_for_stopos.drop("rate", inplace=True, axis=1)
    output_for_stopos.to_csv("output.csv", index=False, sep=" ")
