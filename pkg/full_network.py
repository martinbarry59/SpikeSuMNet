import torch
import memnet_utils as utils
import tqdm
import matplotlib.pyplot as plt
import os
from importlib import reload
import fasNet_module as fasnet
import disinhibitory_module as dm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MemNet(object):

    def __init__(
        self,
        param_networks,
    ):
        """
        Initialize FasNet class
        params: Dictionnary including all network params. More details in
            results/Set_mFasNet_params.ipynb
        """

        # Environment properties
        self.n_memory = param_networks["n_memory"]
        self.number_rooms = param_networks["fasnet_module"]["number_rooms"]
        self.time_good_maze = 0
        self.plot = param_networks['plot']

        # Simulation properties
        self.stop = False
        self.print = param_networks['print']

        # Nework properties
        self.T_pres = param_networks["T_pres"]
        self.batch_size = param_networks['batch_size']
        self.input_neurons = param_networks["fasnet_module"]["input_neurons"]
        self.states = torch.arange(self.input_neurons)
        self.states = self.states.reshape(
            [self.number_rooms, int(float(self.input_neurons) / self.number_rooms)]
        )

        # Neuron Model
        self.Poisson_rate = param_networks["fasnet_module"]["Poisson_rate"]
        self.Poisson_rate_error = param_networks["fasnet_module"]["Poisson_rate_error"]
        self.len_epsc = param_networks["fasnet_module"]["l"]

        # Network layers
        self.fasnet_module_list = []
        self.dishinibitory_module_list = []

        # Add information to parameters for other modules
        param_networks["fasnet_module"]["states"] = self.states
        param_networks["fasnet_module"]["n_memory"] = self.n_memory
        param_networks["selector_module"]["n_memory"] = self.n_memory
        param_networks["fasnet_module"]['batch_size'] = self.batch_size
        param_networks["selector_module"]['batch_size'] = self.batch_size

        # Create Hidden layers (FasNet and selector)
        self.fasnet_module = fasnet.fasNet(param_networks["fasnet_module"])
        self.dishinibitory_module = dm.Disinhibitory_module(
            param_networks["selector_module"])

        self.initiate_info()

    def optimisation_stopping_criterium(self, criterium):
        """
        Determine whether a simulation can be considered as failed or not.

        param criterium: Dictionary with all possible stoping criterions

        return 0 if not failed if if simulations failed
        """
        if (criterium['wrong_change_point'] >
                50 or criterium['before_detecting_cp'] > 50) and self.n_memory > 1:
            criterium['stop'] = True
            return 1
        criterium['last_change_point'] += 1
        if not criterium['cp_detected']:
            criterium['before_detecting_cp'] += 1

        return 0

    def estimate_active_memory(self):
        """
        Readout most active selector module as active memory

        return one_hot version of the active memory and its integer value
        """
        active_memory_activity = self.dishinibitory_module.active_memory_activity
        y = torch.argmax(active_memory_activity, axis=-1)
        one_hot = torch.nn.functional.one_hot(y, num_classes=self.n_memory)
        return one_hot, y

    def initiate_info(self):
        """
        Dic info is a dictionnary saving intersting / to plot data of a simulation
        """
        self.info = {}
        self.info["error"] = torch.zeros(self.batch_size, 1)
        self.info["estimated_mazes"] = torch.zeros(self.batch_size, 1)

    def create_spike_train(self, state=10, neurons=0):
        """
        Simulate Poisson drawing of T_pres time steps for spikes of input neurons

        param state: number of possible different states

        param neurons: number of input neurons

        return: spikes time indices
        """
        proba = torch.zeros((2, neurons, self.T_pres)) + \
            self.Poisson_rate_error
        proba[:, self.states[state], :] = self.Poisson_rate
        spikes = 1.0 * (proba > torch.rand(2, neurons, self.T_pres))

        return spikes

    def plot_network(self, epoch):
        """
        Call the plot function of the different modules`
        """
        if self.plot:
            self.fasnet_module.plot_network()
            if self.n_memory>1:
                self.dishinibitory_module.plot_network()

    def step_info(
        self,
        old_memory,
        estimated_active_memory,
        epoch,
        maze,
        simulation,
        criteria,
        b
    ):
        """
        Print function for inspecting the network state

        param old_memory: previously active memory

        param  estimated_active_memory: newly active memory

        param epoch: Presentation time

        param maze:  True maze

        param simulation: all simulation information

        param criteria: stopping conditions for all batch

        param b: current batch index
        """
        error_array = self.fasnet_module.info['error'][-1]
        plot_times = [1, 500, 1000, 1500, simulation["epochs"] - 1]
        if torch.argmin(error_array) == estimated_active_memory:
            self.time_good_maze += 1
        criteria[b]['count'][estimated_active_memory] += 1
        if old_memory != estimated_active_memory and epoch > 1:
            shape = self.dishinibitory_module.WTA_matrix.shape
            if self.print:
                print(
                    'Simulation N', b + 1,
                    ": Change-point detected. Activating memory : ",
                    estimated_active_memory.item(),
                )
                if maze not in criteria[b]['observed_mazes'] and estimated_active_memory not in criteria[b]['activated_memory']:
                    print("First maze encounter and memory never used")
                elif torch.eq(torch.argmin(error_array[b]), estimated_active_memory):
                    print("Successfully retrieve memory")
                else:
                    print("Wrong memory retrieved")
            if maze not in criteria[b]['observed_mazes']:
                criteria[b]['observed_mazes'] = torch.cat(
                    (criteria[b]['observed_mazes'], torch.Tensor([simulation["maze"][epoch]]))).detach()
            if estimated_active_memory not in criteria[b]['activated_memory']:
                criteria[b]['activated_memory'] = torch.cat(
                    (criteria[b]['activated_memory'], torch.unsqueeze(
                        estimated_active_memory, 0).float())).detach()
            if criteria[b]['cp_detected'] == False or epoch < 10:
                criteria[b]['time_to_detect_cp'] = torch.cat((criteria[b]['time_to_detect_cp'], torch.Tensor([
                                                             criteria[b]['before_detecting_cp']]))).detach()

            criteria[b]['cp_detected'] = True
            old_memory = estimated_active_memory
            if criteria[b]['last_change_point'] > 30:
                criteria[b]['wrong_change_point'] += 1
        if self.batch_size == 1 and epoch in plot_times:
            print('plotting')
            self.plot_network(epoch)

        return old_memory


def batch_spikes(simulation, epoch, network):
    """
    Creation of input spikes following Poisson process

    param epoch: Current presentation time

    param network: Take a full mSpikeSumNet as input

    returns spike train for both Buffer and Observation population and  current maze
    """
    old_room, new_room, maze = (
        simulation["rooms"][epoch],
        simulation["rooms"][epoch + 1],
        simulation["maze"][epoch + 1],
    )
    buffer_spikes = network.create_spike_train(
        old_room, neurons=network.input_neurons
    )
    
    observation_spikes = network.create_spike_train(
        new_room, neurons=network.input_neurons
    )
    return buffer_spikes, observation_spikes, maze


def full_simulation(simulations, network):
    """
    Running/updating the a given network for a full simulation time

    param simulations: list of dictionnaries with Batch_size simulation with all information (Transitions,switch point etc...)

    param network: Takes a mspikeSumNet network as input

    return: stopping Criteria of all simulations
    """

    disable = 1 - network.plot
    old_memory = 0
    # torch.cuda.empty_cache()

    # Simulation initialisation
    errors_updates = torch.zeros(
        network.n_memory * (simulations[0]["epochs"] - 1)).to(device)
    updates = torch.zeros(
        network.n_memory * (simulations[0]["epochs"] - 1)).to(device)
    EPSC_buffer_decay = torch.zeros(
        (network.batch_size, 2, network.input_neurons)).to(device)
    EPSC_observation_decay = torch.zeros(
        (network.batch_size, 2, network.input_neurons)).to(device)
    memory_inhibitory_input = torch.zeros_like(
        network.fasnet_module.h).to(device)
    observation_spikes = torch.zeros(
        network.batch_size,
        2,
        network.input_neurons,
        network.T_pres).to(device)
    buffer_spikes = torch.zeros(
        network.batch_size,
        2,
        network.input_neurons,
        network.T_pres).to(device)
    mazes = torch.zeros(network.batch_size).to(device)
    Transitions = torch.zeros(
        (network.batch_size,
         network.n_memory,
         network.number_rooms,
         network.number_rooms)).to(device)
    old_memory = -1 * torch.ones(network.batch_size).to(device)
    criterium = {'last_change_point': 0,
                 'before_detecting_cp': 0,
                 'wrong_change_point': 0,
                 'time_to_detect_cp': torch.Tensor().to(device),
                 'observed_mazes': torch.Tensor().to(device),
                 'cp_detected': True,
                 'stop': False,
                 'count': torch.zeros(network.n_memory),
                 'activated_memory': torch.Tensor().to(device), }

    criteria = [criterium.copy() for _ in range(network.batch_size)]

    # Running every stimulus presentation
    for epoch in tqdm.tqdm(range(simulations[0]["epochs"]), disable=disable):
        for b in range(network.batch_size):
            buffer, obs, maze = batch_spikes(simulations[b], epoch, network)
            observation_spikes[b] = obs.clone()

            buffer_spikes[b] = buffer.clone()
            mazes[b] = maze
            Transitions[b] = torch.unsqueeze(
                simulations[b]["transitions"][maze], 0).repeat(
                network.n_memory, 1, 1)
            if criteria[b]['stop'] == False and network.optimisation_stopping_criterium(
                    criteria[b]):
                print('Simulation N {0}: failed'.format(b + 1))
                print(
                    'percentage of failure:',
                    round(
                        100 *
                        torch.mean(
                            torch.Tensor(
                                [
                                    criteria[b]['stop'] for b in range(
                                        network.batch_size)])).item(),
                        2))

            estimated_active_memory_one_hot, estimated_active_memory = network.estimate_active_memory()

            # True change-point in the simulation
            if network.batch_size >= 1:
                if epoch in simulations[b]["change_points"].keys():

                    if network.print:
                        print(
                            "Simulation N {2}: Epoch {0} -> Moving to maze {1}".format(
                                epoch, simulations[b]["change_points"][epoch], b + 1
                            )
                        )
                    if epoch > 0:
                        criteria[b]['last_change_point'] = 0
                        criteria[b]['before_detecting_cp'] = 0
                        criteria[b]['cp_detected'] = False

                if epoch > 0:

                    old_memory[b] = network.step_info(
                        old_memory[b],
                        estimated_active_memory[b],
                        epoch,
                        mazes[b],
                        simulations[b],
                        criteria,
                        b
                    )
        network.fasnet_module.save_prediction(
            Transitions
        )
        error_array = network.fasnet_module.info['error'][-1]
        active_error = torch.masked_select(
            error_array, estimated_active_memory_one_hot.ge(0.1))
        network.info["error"] = torch.cat(
            (network.info["error"], torch.unsqueeze(
                active_error, 1)), dim=-1).detach()
        if torch.mean(torch.Tensor([criteria[b]['stop']
                      for b in range(network.batch_size)])) > 0.5:
            print('TOO many simulations failed end of simulation')
            return criteria

        learning = True
        ## spike plot few steps ##

        EPSC_buffer_decay *= 0
        EPSC_observation_decay *= 0
        memory_inhibitory_input *= 0
        network.fasnet_module.clear_spike_train()
        network.dishinibitory_module.clear_spike_train()
        mean_update = []
        mean_input = []
        for t in range(network.T_pres):
            
            EPSC_buffer, EPSC_buffer_decay = utils.square_EPSC(
                EPSC_buffer_decay,
                network.len_epsc,
                buffer_spikes[:, :, :, t].clone(),
            )
            EPSC_observation, EPSC_observation_decay = utils.square_EPSC(
                EPSC_observation_decay,
                network.len_epsc,
                observation_spikes[:, :, :, t].clone(),
            )
            # FasNet step
            weight_update, commitement_update = network.fasnet_module.forward(
                EPSC_buffer,
                EPSC_observation,
                memory_inhibitory_input,
                learning=learning,
            )

            dishin_input = network.fasnet_module.output.clone()
            # does the WTA only if more than one memory
            # MSM step
            network.dishinibitory_module.forward(
                dishin_input, commitement_update, learning=learning
            )
            memory_inhibitory_input = network.dishinibitory_module.memory_output.clone()

            mean_update += [weight_update.detach()]
            # save output spikes of module to send to others
            mean_input += [network.dishinibitory_module.input.detach()]
#             mean_input += [network.fasnet_module.input.detach()]
            # mean_input += [torch.mean(network.fasnet_module.ratio)]

        if epoch > 0 and network.batch_size == 1:

            # print(torch.mean(torch.cat(mean_input),axis=0))

            errors_updates[network.n_memory * (epoch - 1):network.n_memory * (
                epoch - 1) + network.n_memory] = network.fasnet_module.info['error'][epoch - 1]
            updates[network.n_memory * (epoch - 1):network.n_memory * (epoch - 1) + network.n_memory] = torch.mean(
                torch.cat(mean_update).view(network.T_pres, network.n_memory))
        network.fasnet_module.filtered_EPSC *= 0
        network.dishinibitory_module.info["WTA"] = [
            network.dishinibitory_module.commitment_matrix]

    network.info['fasnet_module'] = network.fasnet_module.info
    network.info['dishinibitory_module'] = network.dishinibitory_module.info
    return criteria
