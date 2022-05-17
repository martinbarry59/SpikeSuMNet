import torch
import network_utils
import tqdm
import SpikeSuM_module as SpikeSuM
import selector_module as sm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def init_agent(simulation_data, network):
    """
    Reset the network after each observation

    simulation_data: dictionaire with necessary information about the simulation step

    network: A SpikeSuM-C network used for the simulation

    return: updated simulation_data
    """
    simulation_data['EPSC_buffer_decay'] *= 0
    simulation_data['EPSC_observation_decay'] *= 0
    network.SpikeSuM_module.EPSC_EI_decay *= 0
    network.SpikeSuM_module.h *= 0
    network.SpikeSuM_module.clear_spike_train()
    network.selector_module.clear_spike_train()
    return simulation_data
def pre_step(simulation_data, network,simulations, epoch):
    """
    simulation update necessary before the actual update (saving data, drawing new steps, checking for failures...)  

    simulation_data: dictionaire with necessary information for network simulation step

    network: A SpikeSuM-C network used for the simulation
    
    simulations: dictionaire containing the mazes information
    
    epoch: The current observation step

    return: updated simulation_data
    """
    if network.output is not None:
            network.f = open(network.output, "a")
    else:
        network.f = None
    estimated_active_memory_one_hot, estimated_active_memory = network.estimate_active_memory()
    for b in range(network.batch_size):
        buffer, obs, maze = network_utils.batch_spikes(simulations[b], epoch, network)
        simulation_data['observation_spikes'][b] = obs.clone()

        simulation_data['buffer_spikes'][b] = buffer.clone()
        simulation_data['mazes'][b] = maze
        simulation_data['Transitions'][b] = torch.unsqueeze(
            simulations[b]["transitions"][maze], 0).repeat(
            network.n_memory, 1, 1)
        if simulation_data['criteria'][b]['stop'] == False and network.optimisation_stopping_criterium(
                simulation_data['criteria'][b]):
            print('Simulation N {0}: failed'.format(b + 1), file=network.f)
            print(
                'percentage of failure:',
                round(
                    100 *
                    torch.mean(
                        torch.Tensor(
                            [
                                simulation_data['criteria'][b]['stop'] for b in range(
                                    network.batch_size)])).item(),
                    2), file=network.f)

        
        
        # Updating criteria after change point
        if epoch in simulations[b]["change_points"].keys():

            if network.print:
                print(
                    "Simulation N {2}: Epoch {0} -> Moving to maze {1}".format(
                        epoch, simulations[b]["change_points"][epoch], b + 1
                    ), file=network.f
                )
            if epoch > 0:
                simulation_data['criteria'][b]['last_change_point'] = 0
                simulation_data['criteria'][b]['before_detecting_cp'] = 0
                simulation_data['criteria'][b]['cp_detected'] = False
        
        # Updating network informations
        if epoch > 0:

            simulation_data['old_memory'][b] = network.step_info(
                simulation_data['old_memory'][b],
                estimated_active_memory[b],
                epoch,
                simulation_data['mazes'][b],
                simulations[b],
                simulation_data['criteria'],
                b
            )
            
    network.SpikeSuM_module.save_prediction(
            simulation_data['Transitions']
        )
    
    error_array = network.SpikeSuM_module.info['error'][-1]
    active_error = torch.masked_select(
        error_array, estimated_active_memory_one_hot.ge(0.1))
    network.info["error"] = torch.cat(
        (network.info["error"], torch.unsqueeze(
            active_error, 1)), dim=-1).detach()
    
    
    if torch.mean(torch.Tensor([simulation_data['criteria'][b]['stop']
                    for b in range(network.batch_size)])) > 0.5:
        print('TOO many simulations failed end of simulation', file=network.f)
        return simulation_data['criteria'], epoch
    
    simulation_data = init_agent(simulation_data, network)
    return simulation_data

def initialise_simulation(simulations, network):
    """
    First instantiation of the **simulation_data** dictionnaire, that will save the network informations during the the simulation

    simulations: dictionaire containing the mazes information

    network: A SpikeSuM-C network used for the simulation

    return: simulation_data
    """
    
    simulation_data = {}
    simulation_data['errors_updates'] = torch.zeros(
        network.n_memory * (simulations[0]["epochs"] - 1)).to(device)
    simulation_data['updates'] = torch.zeros(
        network.n_memory * (simulations[0]["epochs"] - 1)).to(device)
    simulation_data['EPSC_buffer_decay'] = torch.zeros(
        (network.batch_size, 2, network.input_neurons)).to(device)
    simulation_data['EPSC_observation_decay'] = torch.zeros(
        (network.batch_size, 2, network.input_neurons)).to(device)
    simulation_data['memory_inhibitory_input'] = torch.zeros_like(network.SpikeSuM_module.h).to(device)
    simulation_data['observation_spikes'] = torch.zeros(
        network.batch_size,
        2,
        network.input_neurons,
        network.T_pres).to(device)
    simulation_data['buffer_spikes'] = torch.zeros(
        network.batch_size,
        2,
        network.input_neurons,
        network.T_pres).to(device)
    simulation_data['mazes'] = torch.zeros(network.batch_size).to(device)
    simulation_data['Transitions'] = torch.zeros(
        (network.batch_size,
         network.n_memory,
         network.number_rooms,
         network.number_rooms)).to(device)
    simulation_data['old_memory'] = -1 * torch.ones(network.batch_size).to(device)
    criterium = {'last_change_point': 0,
                 'before_detecting_cp': 0,
                 'wrong_change_point': 0,
                 'time_to_detect_cp': torch.Tensor().to(device),
                 'observed_mazes': torch.Tensor().to(device),
                 'cp_detected': True,
                 'stop': False,
                 'count': torch.zeros(network.n_memory),
                 'activated_memory': torch.Tensor().to(device), }

    # Dictionnary saving the stopping criterium of each simulation in batch dimension
    simulation_data['criteria'] = [criterium.copy() for _ in range(network.batch_size)]
    return simulation_data
def print_presentation_summary(means, simulation_data, network, simulations, epoch):
    """
    print information of one observation step (used for debugging) 

    means: array composed of information of network spikes for T_pres
    
    simulation_data: dictionaire with necessary information for network simulation step

    network: A SpikeSuM-C network used for the simulation
    
    simulations: dictionaire containing the mazes information
    
    epoch: The current observation step

    return: None
    """
    if epoch > 0 and network.batch_size <= 1 and network.print:
        for context in range(network.batch_size):
            print(simulations[context]["rooms"][epoch].item(), simulations[context]["rooms"][epoch+1].item(),
                torch.mean(torch.cat(means['mean_input']),axis=0),
                'Threshold: ',torch.mean(torch.cat(means['mean_input_']),axis=0))
        
def presentation_to_agent(simulation_data, network, simulations, epoch):
    """
    Presentation of stimulus to the network and updates
    
    simulation_data: dictionaire with necessary information for network simulation step

    network: A SpikeSuM-C network used for the simulation
    
    simulations: dictionaire containing the mazes information
    
    epoch: The current observation step

    return: updated simulation_data
    """
    learning = True
    means = {}
    means['mean_update'] = []
    means['mean_input'] = []
    means['mean_input_'] = []
    for t in range(network.T_pres):
            
            EPSC_buffer, simulation_data['EPSC_buffer_decay'] = network_utils.square_EPSC(
                simulation_data['EPSC_buffer_decay'],
                network.len_epsc,
                simulation_data['buffer_spikes'][:, :, :, t].clone(),
            )
            EPSC_observation, simulation_data['EPSC_observation_decay'] = network_utils.square_EPSC(
                simulation_data['EPSC_observation_decay'],
                network.len_epsc,
                simulation_data['observation_spikes'][:, :, :, t].clone(),
            )
            # SpikeSuM step
            weight_update, commitement_update = network.SpikeSuM_module.forward(
                EPSC_buffer,
                EPSC_observation,
                simulation_data['memory_inhibitory_input'],
                learning=learning,
            )

            dishin_input = network.SpikeSuM_module.output.clone()
            # does the WTA only if more than one memory
            # MSM step
            network.selector_module.forward(
                dishin_input, commitement_update, learning=learning
            )
            simulation_data['memory_inhibitory_input'] = network.selector_module.memory_output.clone()
            means['mean_update'] += [weight_update.detach()]
            # save output spikes of module to send to others
            means['mean_input'] += [network.selector_module.input.detach()]
            means['mean_input_'] += [network.selector_module.meanput.detach()]
    print_presentation_summary(means, simulation_data, network, simulations, epoch)
    network.SpikeSuM_module.filtered_EPSC *= 0
    
    network.selector_module.info["WTA"] = [
    network.selector_module.commitment_matrix]
    if epoch > 0 and network.batch_size == 1:
        simulation_data['errors_updates'][network.n_memory * (epoch - 1):network.n_memory * (
            epoch - 1) + network.n_memory] = network.SpikeSuM_module.info['error'][epoch - 1]
        simulation_data['updates'][network.n_memory * (epoch - 1):network.n_memory * (epoch - 1) + network.n_memory] = torch.mean(
            torch.cat(means['mean_update']).view(network.T_pres, network.n_memory))
    return simulation_data
def run_simulation(simulations, network):
    """
    Running/updating the a given network for a full simulation time

    param simulations: list of dictionnaries with Batch_size simulation with all information (Transitions,switch point etc...)

    param network: Takes a mspikeSumNet network as input

    return: stopping Criteria of all simulations
    """
    
    disable = 1 - network.plot

    # Simulation initialisation

    simulation_data = initialise_simulation(simulations, network)

    #Running every stimulus presentation
    for epoch in tqdm.tqdm(range(simulations[0]["epochs"]), disable=disable):

        simulation_data = pre_step(simulation_data, network,simulations, epoch)
        
        simulation_data = presentation_to_agent(simulation_data, network, simulations, epoch)    

        if  network.f is not None:
            network.f.close()
    if network.batch_size == 1 and network.SpikeSuM_module.tosave is not None:
        for key in ['','_neg','_pos']:
            network.SpikeSuM_module.info['effective update'+key] = torch.cat(network.SpikeSuM_module.info['effective update'+key]).flatten()
            network.SpikeSuM_module.info['prediction error'+key] = torch.cat(network.SpikeSuM_module.info['prediction error'+key]).flatten()
    network.info['SpikeSuM_module'] = network.SpikeSuM_module.info
    network.info['selector_module'] = network.selector_module.info
    return simulation_data['criteria'], epoch
