import copy
import os
import pickle
import time
from argparse import ArgumentParser

import netsquid as ns
import pandas as pd
import yaml
from netsquid.nodes import Connection
from netsquid.nodes import Node
from netsquid.util import DataCollector
from netsquid_driver.EGP import EGPService
from netsquid_magic.magic_distributor import DoubleClickMagicDistributor
from netsquid_magic.magic_distributor import SingleClickMagicDistributor
from netsquid_netconf.netconf import ComponentBuilder
from netsquid_netconf.netconf import Loader
from netsquid_netconf.netconf import netconf_generator
from netsquid_nv.magic_distributor import NVDoubleClickMagicDistributor
from netsquid_nv.magic_distributor import NVSingleClickMagicDistributor
from netsquid_physlayer.classical_connection import ClassicalConnectionWithLength
from netsquid_physlayer.heralded_connection import HeraldedConnection
from netsquid_physlayer.heralded_connection import MiddleHeraldedConnection
from netsquid_simulationtools.repchain_data_plot import plot_teleportation
from netsquid_simulationtools.repchain_data_process import process_data_duration
from netsquid_simulationtools.repchain_data_process import process_data_teleportation_fidelity
from netsquid_simulationtools.repchain_data_process import process_repchain_dataframe_holder
from netsquid_simulationtools.repchain_dataframe_holder import RepchainDataFrameHolder
from nlblueprint.analytics.effect_of_time_windows import coincidence_probability
from nlblueprint.analytics.effect_of_time_windows import coincidence_probability_dark_count_and_photon
from nlblueprint.analytics.effect_of_time_windows import coincidence_probability_two_dark_counts
from nlblueprint.analytics.effect_of_time_windows import innsbruck_emission_time_decay_param
from nlblueprint.analytics.effect_of_time_windows import innsbruck_t_win
from nlblueprint.analytics.effect_of_time_windows import innsbruck_wave_function_decay_param
from nlblueprint.analytics.effect_of_time_windows import visibility
from nlblueprint.egp_datacollector import EGPDataCollectorState
from nlblueprint.processing_nodes.nodes_with_drivers import AbstractNodeWithDriver
from nlblueprint.processing_nodes.nodes_with_drivers import DepolarizingNodeWithDriver
from nlblueprint.processing_nodes.nodes_with_drivers import NVNodeWithDriver
from nlblueprint.processing_nodes.nodes_with_drivers import TINodeWithDriver
from pydynaa.core import EventExpression
from qlink_interface import ReqCreateAndKeep
from qlink_interface import ResCreateAndKeep
# from netsquid_ae.ae_classes import EndNode, RepeaterNode, MessagingConnection, QKDNode
# from nlblueprint.atomic_ensembles.ae_rb_tm_node_setup import RbTmRepeaterNode
# from nlblueprint.atomic_ensembles.rb_tm_chain_setup import _add_rb_protocols_and_magic
# from netsquid_ae.ae_classes import HeraldedConnection as AEHeraldedConnection
# from netsquid_ae.ae_chain_setup import create_qkd_application, _add_protocols_and_magic
# from nlblueprint.control_layer.magic_EGP import MagicEGP
# from nlblueprint.atomic_ensembles.ae_magic_link_layer import AEMagicLinkLayerProtocolWithSignalling
# from nlblueprint.atomic_ensembles.magic.forced_magic_factory import generate_forced_magic

distance_delft_eindhoven = 226.5
distributor_name_to_class = {"double_click": DoubleClickMagicDistributor,
                             "single_click": SingleClickMagicDistributor,
                             "single_click_nv": NVSingleClickMagicDistributor,
                             "double_click_nv": NVDoubleClickMagicDistributor}

"""
This script takes as input a configuration file and a parameter file. You can find examples of these for each of the
platforms we simulate in the same folder. It parses these files using netconf and runs an end-to-end entanglement
generation experiment using a link layer protocol. It then stores this data in a RepChainDataFrameHolder object and
saves it in a pickle file. The script is capable of detecting if a parameter is being varied in a configuration file
and, using the netconf snippet, run the simulation for each of the values of this parameter.

Through optional input arguments one can define the path where the results should be saved, how many runs per data
point to simulate, the name of the saved file and whether the simulation results should be plotted.
"""


def reset(network):
    """Stop protocols and reset components to get ready for next simulation run."""

    for node in network.nodes.values():
        for service_instantiation in node.driver.services.values():
            service_instantiation.reset()
        node.reset()
    for connection in network.connections.values():
        connection.reset()


def implement_magic(network, config):
    """Add magic distributors to a physical network.

    To every connection in the network a magic distributor is added as attribute.
    The distributor type must be defined in the configuration file for each heralded connection, as a property of name
    "distributor". The supported distributor types can be found in the `distributor_name_to_class` dictionary, where the
    keys are the arguments that should be defined in the configuration file, and the values are the actual distributors.
    The magic distributor is given the connection as argument upon construction (so that parameters to be used in magic
    can be read from the connection).

    Parameters
    ----------
    network : :class:`netsquid.nodes.network.Network`
        Network that the network distributors should be added to.
    config : dict
        Dictionary holding component names and their properties.

    """
    distributors = {}
    for name, connection in config["components"].items():
        if "heralded_connection" in connection["type"]:
            try:
                distributors[name] = connection["properties"]["distributor"]
            except KeyError:
                raise KeyError("Magic distributor not defined for {}".format(name))

    for connection in network.connections.values():
        if isinstance(connection, HeraldedConnection):
            nodes = [port.connected_port.component for port in connection.ports.values()]
            try:
                magic_distributor = distributor_name_to_class[distributors[connection.name]](
                    nodes=nodes, heralded_connection=connection)
            except KeyError:
                raise KeyError("{} is not a supported distributor type.".format(distributors[connection.name]))
            connection.magic_distributor = magic_distributor


def setup_networks(config_file_name, param_file_name):
    """
    Set up network(s) according to configuration file

    Parameters
    ----------
    config_file_name : str
        Name of configuration file.
    param_file_name : str
        Name of parameter file.

    Returns
    -------
    generator : generator
        Generator yielding network configurations
    ae : bool
        True if simulating atomic ensemble hardware, False otherwise
    number_nodes : int
        Number of nodes in the network
    """

    # add required components to ComponentBuilder
    ComponentBuilder.add_type(name="abstract_node", new_type=AbstractNodeWithDriver)
    ComponentBuilder.add_type(name="ti_node", new_type=TINodeWithDriver)
    ComponentBuilder.add_type(name="nv_node", new_type=NVNodeWithDriver)
    ComponentBuilder.add_type(name="depolarizing_node", new_type=DepolarizingNodeWithDriver)
    ComponentBuilder.add_type(name="mid_heralded_connection", new_type=MiddleHeraldedConnection)
    ComponentBuilder.add_type(name="heralded_connection", new_type=HeraldedConnection)
    ComponentBuilder.add_type(name="classical_connection", new_type=ClassicalConnectionWithLength)
    # ComponentBuilder.add_type(name="AEEndNode", new_type=EndNode)
    # ComponentBuilder.add_type(name="AERepeaterNode", new_type=RepeaterNode)
    # ComponentBuilder.add_type(name="RbTmRepeaterNode", new_type=RbTmRepeaterNode)
    # ComponentBuilder.add_type(name="AEHeraldedConnection", new_type=AEHeraldedConnection)
    # ComponentBuilder.add_type(name="AEMessagingConnection", new_type=MessagingConnection)

    generator = netconf_generator(config_file_name)
    objects, config = next(generator)

    ae = False
    # for component in objects["components"].values():
    #     if isinstance(component, EndNode):
    #         ae = True
    #         break

    number_nodes = _get_number_nodes(objects["components"], ae)

    # if ae and param_file_name is not None:
    #     # read sim_params from paramfile
    #     with open(param_file_name, "r") as stream:
    #         sim_params = yaml.load(stream, Loader=Loader)
    #
    #     if sim_params["magic"] == "forced":
    #         generate_forced_magic(sim_params)
    #         ns.sim_reset()
    #
    #         # dump sim_params with updated sample_file name
    #         with open(param_file_name, "w") as stream:
    #             yaml.dump(sim_params, stream)
    # elif ae and param_file_name is None:
    #     raise RuntimeError("Atomic ensemble implementation requires parameter file to be specified.")

    generator = netconf_generator(config_file_name)
    return generator, ae, number_nodes


def _get_number_nodes(components, ae):
    """Computes the number of nodes in the chain from the generated components.

    In the case of processing nodes, this is done in a straightforward manner by counting the components of the Node
    type. For AEs, one must check if a double chain is being used by comparing the number of heralded connections
    (which is half of the total number of connections, since the latter includes classical connections) to the number
    of node components. If a double chain is being used, the actual number of nodes is half of the number found
    by counting components of the Node type.

    Parameters
    ----------
    components : dict
        Dictionary holding components generated by netconf from configuration file
    ae : bool
        True if atomic ensemble based repeaters are being simulated

    Returns
    -------
    number_nodes : int
        Number of nodes in chain being simulated

    """
    number_nodes = 0
    for component in components.values():
        if isinstance(component, Node):
            number_nodes += 1

    if ae:
        number_connections = 0
        for component in components.values():
            if isinstance(component, Connection):
                number_connections += 1
        number_connections /= 2

        if number_connections == number_nodes - 2:
            number_nodes //= 2

    return number_nodes


def find_varied_param(generator):
    """
    Determines which (if any) parameter is being varied in the simulation by comparing the configurations generated
    by netconf.

    Parameters
    ----------
    generator : generator
        Generator of network and configurations generated by netconf

    Returns
    -------
    key : str or None
        Name of object whose parameter is being varied
    property_key : str or None
        Name of parameter being varied
    """

    configs = []
    for _, config in generator:
        configs_temp = {}
        for key, value in config["components"].items():
            configs_temp[key] = copy.deepcopy(value["properties"])
        configs.append(configs_temp)

        if len(configs) == 2:
            break

    if len(configs) < 2:
        return None, None

    for key, value in configs[0].items():
        if configs[1][key] != value:
            for property_key, property_value in configs[1][key].items():
                if property_value != configs[0][key][property_key]:
                    return key, property_key


def run_simulation(egp_services, request_type, response_type, data_collector, n_runs, experiment_specific_arguments={}):
    """
    Places a request for a given number of pairs to be created and kept, runs the simulation and
    collects the resulting data.

    Parameters
    ----------
    egp_services : list
        List holding EGP end node services
    request_type : :class:`qlink_interface.ReqCreateBase`
        Type of request to be submitted.
    response_type : :class:`qlink_interface.ResCreateBase`
        Type of response corresponding to the submitted request
    data_collector : :class:`nlblueprint.atomic_ensembles.egp_datacollector.EGPDataCollector`
        Data collector to use
    n_runs : int
        Number of pairs to generate and measure
    experiment_specific_arguments : dict
        Dictionary holding extra experiment-specific arguments e.g. measurement bases

    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe holding simulation results
    """

    alice_responds = EventExpression(source=egp_services[0],
                                     event_type=egp_services[0].signals.get(response_type.__name__))
    bob_responds = EventExpression(source=egp_services[1],
                                   event_type=egp_services[1].signals.get(response_type.__name__))

    both_nodes_respond = alice_responds & bob_responds
    egp_datacollector = DataCollector(data_collector(start_time=ns.sim_time()))
    egp_datacollector.collect_on(both_nodes_respond)

    local = egp_services[0]
    remote = egp_services[1]
    request = request_type(remote_node_id=remote.node.ID,
                           number=n_runs,
                           **experiment_specific_arguments)

    local.put(request)

    # run simulation
    ns.sim_run()

    return egp_datacollector.dataframe


def magic_and_protocols(network, config, ae, sim_params=None):
    """
    Sets up protocols and magic distributors for simulation.

    Parameters
    ----------
    network : :class:`netsquid.nodes.network.Network`
        Network to be simulated
    ae : bool
        If True, simulating atomic ensemble hardware. Otherwise, simulating processing nodes
    sim_params : dict (optional)
        Dictionary holding simulation parameters

    Returns
    -------
    list
        List holding EGP end node services
    """
    # if ae:
    #     egp_services = magic_and_protocols_ae(sim_params, network)
    # else:
    implement_magic(network, config)
    _start_all_services(network)
    end_nodes = [node for node in network.nodes.values() if len(node.ports) == 2]
    egp_services = [end_node.driver[EGPService] for end_node in end_nodes]
    return egp_services


def _start_all_services(network):
    """
    Starts all services running on the network's nodes.

    Parameters
    ----------
    network : :class:`netsquid.nodes.network.Network`
        Network to be simulated

    """

    for node in network.nodes.values():
        node.driver.start_all_services()


# def magic_and_protocols_ae(sim_params, network):
#     """
#     Sets up protocols and magic distributors for AE simulation.
#
#     Parameters
#     ----------
#     sim_params : dict
#         Dictionary holding simulation parameters
#     network : :class:`netsquid.nodes.network.Network`
#         Network to be simulated
#
#     Returns
#     -------
#     list
#         List holding AE EGP end node services
#     """
#     try:
#         _ = sim_params["cooperativity"]     # if cooperativity is in sim_params run Rb simulation
#         protocols, magic_prot, magic_dist = _add_rb_protocols_and_magic(network, sim_params)
#     except KeyError:
#         protocols, magic_prot, magic_dist = _add_protocols_and_magic(network, sim_params)
#
#     # add QKD application
#     det_protos = create_qkd_application(network=network, sim_params=sim_params, measure_directly=True,
#                                         initial_measurement_basis="X")
#     protocols.extend(det_protos)
#
#     # Start Protocols
#     # =================
#     for proto in protocols:
#         proto.start()
#
#     # set up EGP Service
#     end_nodes = [node for node in network.nodes.values() if isinstance(node, QKDNode)]
#     ae_link_layer = AEMagicLinkLayerProtocolWithSignalling(nodes=end_nodes, protocols=protocols[-2:],
#                                                            magic_distributor=magic_dist)
#
#     ae_egp_service_alice = MagicEGP(node=end_nodes[0], magic_link_layer_protocol=ae_link_layer)
#     ae_egp_service_bob = MagicEGP(node=end_nodes[1], magic_link_layer_protocol=ae_link_layer)
#
#     ae_egp_services = [ae_egp_service_alice, ae_egp_service_bob]
#
#     ae_link_layer.start()
#     for serv in ae_egp_services:
#         serv.start()
#     return ae_egp_services


def save_data(meas_holder, args):
    """
    Saves the repchain dataframe holder with the simulation data to the specified output path.

    Parameters
    ----------
    meas_holder : :class:`netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`
        RepchainDataFrameHolder with collected simulation data.
    args : :class: 'argparse.Namespace`
        Holds parsed input arguments.
    """

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    # save as csv (and make sure nothing is overwritten)

    saved = False

    while not saved:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        arg_dict = copy.deepcopy(vars(args))
        arg_dict["configfile"] = arg_dict["configfile"].split("/")[-1]
        arg_dict.pop("plot")
        if isinstance(arg_dict["paramfile"], str):
            arg_dict["paramfile"] = arg_dict["paramfile"].split("/")[-1]
        arg_dict_no_none_values = {k: v for k, v in arg_dict.items() if v is not None}
        argstring = str(arg_dict_no_none_values).translate(
            {**{ord(i): None for i in "{' }"}, **{ord(j): "_" for j in ":,"}})

        filename = args.output_path + "/" + argstring + "_" + timestr

        if not os.path.isfile(filename):
            with open(filename + ".pickle", 'wb') as handle:
                pickle.dump(meas_holder, handle, protocol=pickle.HIGHEST_PROTOCOL)
            saved = True


def plot_data(meas_holder):
    """
    Plots quantum-teleportation data using the simulation tools snippet.

    Parameters
    ----------
    meas_holder : :class:`netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`
        RepchainDataFrameHolder with collected state data.

    """
    if len(meas_holder.varied_parameters) != 1:
        raise ValueError("Can only plot for data with exactly one varied parameter. "
                         f"This data has the following varied parameters: {meas_holder.varied_parameters}")
    [varied_param] = meas_holder.varied_parameters
    processed_data = process_repchain_dataframe_holder(repchain_dataframe_holder=meas_holder,
                                                       processing_functions=[process_data_duration,
                                                                             process_data_teleportation_fidelity])
    processed_data.to_csv("output.csv")
    plot_teleportation(filename="output.csv", scan_param_name=varied_param, scan_param_label=varied_param)


def collect_state_data(generator, n_runs, ae, sim_params, varied_param,
                       varied_object, number_nodes, suppress_output=False):
    """
    Runs simulation and collects data for each network configuration. Stores data in RepchainDataFrameHolder in format
    that allows for plotting at later point.

    Parameters
    ----------
    generator : generator
        Generator of network configurations
    n_runs : int
        Number of runs per data point
    ae : bool
        True if simulating AE hardware
    sim_params : dict or None
        Dictionary holding simulation parameters
    varied_param : str or None
        Name of parameter being varied
    varied_object : str or None
        Name of object whose parameter is being varied
    number_nodes : int
        Number of nodes in chain being simulated
    suppress_output : bool
        If true, status print statements are suppressed.

    Returns
    -------
    meas_holder : :class:`netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`
        RepchainDataFrameHolder with collected simulation data.

    """
    if ae:
        ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.SPARSEDM)
    else:
        ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    if not suppress_output:
        if varied_param is None:
            print("Simulating", n_runs, "runs")
        else:
            print("Simulating", n_runs, "runs for each value of", varied_param)

    data = []
    for objects, config in generator:
        network = objects["network"]
        egp_services = magic_and_protocols(network, config, ae, sim_params)
        start_time_simulation = time.time()
        data_one_configuration = run_simulation(egp_services=egp_services,
                                                request_type=ReqCreateAndKeep,
                                                response_type=ResCreateAndKeep,
                                                data_collector=EGPDataCollectorState,
                                                n_runs=n_runs)
        if varied_param is not None:
            param_value = config["components"][varied_object]["properties"][varied_param]
            data_one_configuration[varied_param] = param_value
        simulation_time = time.time() - start_time_simulation
        if not suppress_output:
            if varied_param is None:
                print(f"Performed {n_runs} runs in {simulation_time:.2e} s")
            else:
                print(f"Performed {n_runs} runs for {varied_param} = {param_value} "
                      f"in {simulation_time:.2e} s")
        data.append(data_one_configuration)

    data = pd.concat(data, ignore_index=True)
    new_data = data.drop(labels=["time_stamp", "entity_name"], axis="columns")

    if sim_params is not None:
        baseline_parameters = sim_params
        baseline_parameters["number_nodes"] = number_nodes
    else:
        baseline_parameters = {"length": distance_delft_eindhoven,
                               "number_nodes": number_nodes}
    print(new_data)
    meas_holder = RepchainDataFrameHolder(baseline_parameters=baseline_parameters, data=new_data, number_of_nodes=2)

    return meas_holder


def run_unified_simulation_state(configfile, paramfile=None, n_runs=10, suppress_output=False):
    """Run unified simulation script.

    Parameters
    ----------
    config_file_name : str
        Name of configuration file.
    param_file_name : str
        Name of parameter file.
    n_runs : int
        Number of runs per data point
    suppress_output : bool
        If true, status print statements are suppressed.

    Returns
    -------
    meas_holder : :class:`netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`
        RepchainDataFrameHolder with collected QKD data.
    varied_param : str
        Name of parameter that is being varied in the simulation.

    """

    generator, ae, number_nodes = setup_networks(configfile, paramfile)
    varied_object, varied_param = find_varied_param(generator)
    generator, ae, _ = setup_networks(configfile, paramfile)

    # iterate over networks read from config file and collect BB84 data
    if paramfile is not None:
        with open(paramfile, "r") as stream:
            sim_params = yaml.load(stream, Loader=Loader)
    else:
        sim_params = None

    repchain_df_holder = collect_state_data(generator, n_runs, ae, sim_params, varied_param, varied_object,
                                            number_nodes, suppress_output=suppress_output)

    return repchain_df_holder, varied_param


def innsbruck_visibility_and_coin_probs(t_coin, improvement_factor_visibility, improvement_factor_coin_prob_ph_ph):
    """Calculate visibility and coincidence probabilities for Innsbruck setup.

    Parameters
    ----------
    t_coin : float
        Coincidence time window ([ns]). This is the maximum time between two detection events.
    improvement_factor_visibility : float
        Improvement factor for visibility. Improvement is performed using root-based method.
    improvement_factor_coin_prob_ph_ph : float
        Improvement factor for coincidence probability between two photons.
        Improvement is performed using root-based method.

    Returns
    -------
    float
        visibility
    float
        coincidence probability for two photons
    float
        coincidence probability for a photon and a dark count
    float
        coincidence probability for two dark counts

    """

    t_coin = t_coin * 1E-3  # convert to microseconds to match Innsbruck data

    if t_coin > innsbruck_t_win:
        raise ValueError("Coincidence window cannot exceed time window.")
    if t_coin < 0:
        raise ValueError("Coincidence window must be a positive number.")

    # calculate quantities from coincidence-window model
    baseline_visibility = visibility(t_coin=t_coin, emission_time_decay_param=innsbruck_emission_time_decay_param,
                                     wave_function_decay_param=innsbruck_wave_function_decay_param,
                                     t_win=innsbruck_t_win)
    baseline_coin_prob_ph_ph = coincidence_probability(t_coin=t_coin,
                                                       emission_time_decay_param=innsbruck_emission_time_decay_param,
                                                       wave_function_decay_param=innsbruck_wave_function_decay_param,
                                                       t_win=innsbruck_t_win, condition_on_within_time_window=True)
    coin_prob_ph_dc = \
        coincidence_probability_dark_count_and_photon(t_coin=t_coin,
                                                      emission_time_decay_param=innsbruck_emission_time_decay_param,
                                                      wave_function_decay_param=innsbruck_wave_function_decay_param,
                                                      t_win=innsbruck_t_win)
    coin_prob_dc_dc = coincidence_probability_two_dark_counts(t_coin=t_coin, t_win=innsbruck_t_win)

    # perform parameter improvements
    vis = baseline_visibility ** (1. / improvement_factor_visibility)
    coin_prob_ph_ph = baseline_coin_prob_ph_ph ** (1. / improvement_factor_coin_prob_ph_ph)

    return vis, coin_prob_ph_ph, coin_prob_ph_dc, coin_prob_dc_dc


# todo: now replaced by `yotse.pre.Parameter.depends_on`
# def _translate_smart_stopos_blueprint(param_file):
#     # load sim parameters from file
#     with open(param_file, "r") as stream:
#         sim_params = yaml.load(stream, Loader=Loader)
#
#     if "cutoff_time" in sim_params and sim_params["cutoff_time"] is not None:
#         sim_params["cutoff_time"] = sim_params["cutoff_time"] * sim_params["carbon_T2"]
#
#     # dump to yaml to be picked up by netconf
#     with open(args.paramfile, "w") as stream:
#         yaml.dump(sim_params, stream)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('configfile', type=str, help="Name of the config file.")
    parser.add_argument('-pf', '--paramfile', required=False, type=str,
                        help="Name of the parameter file. Only required as a command line argument for AE simulations.")
    parser.add_argument('-n', '--n_runs', required=False, type=int, default=10,
                        help="Number of runs per configuration. If none is provided, defaults to 10.")
    parser.add_argument('--output_path', required=False, type=str, default="raw_data",
                        help="Path relative to local directory where simulation results should be saved.")
    parser.add_argument('--filebasename', required=False, type=str, help="Name of the file to store results in.")
    parser.add_argument('--plot', dest="plot", action="store_true",
                        help="Plot the simulation results. Currently not available.")

    args, unknown = parser.parse_known_args()
    # _translate_smart_stopos_blueprint(args.paramfile)
    repchain_df_holder, varied_param = run_unified_simulation_state(configfile=args.configfile,
                                                                    paramfile=args.paramfile,
                                                                    n_runs=args.n_runs,
                                                                    suppress_output=False)
    save_data(repchain_df_holder, args)
    if args.plot:
        plot_data(repchain_df_holder, varied_param)
