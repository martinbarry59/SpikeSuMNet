import torch
import plot_utils
import tqdm
import matplotlib.pyplot as plt
import os
from importlib import reload
import SpikeSuM_module as SpikeSuM
import selector_module as sm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SpikeSuMC(object):

    def __init__(
        self,
        param_networks,
        outputfile = 'logs.txt'
    ):
        """
        Initialize SpikeSuM class
        params: Dictionnary including all network params. More details in
            results/Set_SpikeSuM-M_params.ipynb
        """
        #plot_utils.nice_print(param_networks)
        self.output = outputfile
        # Environment properties
        self.n_memory = param_networks["n_memory"]
        self.number_rooms = param_networks["SpikeSuM_module"]["number_rooms"]
        self.time_good_maze = 0
        self.plot = param_networks['plot']
        # Simulation properties
        self.stop = False
        self.print = param_networks['print']

        # Nework properties
        self.T_pres = param_networks["T_pres"]
        self.batch_size = param_networks['batch_size']
        self.input_neurons = param_networks["SpikeSuM_module"]["input_neurons"]
        self.states = torch.arange(self.input_neurons)
        self.states = self.states.reshape(
            [self.number_rooms, int(float(self.input_neurons) / self.number_rooms)]
        )

        # Neuron Model
        self.Poisson_rate = param_networks["SpikeSuM_module"]["Poisson_rate"]
        self.Poisson_rate_error = param_networks["SpikeSuM_module"]["Poisson_rate_error"]
        self.len_epsc = param_networks["SpikeSuM_module"]["l"]


        # Add information to parameters for other modules
        param_networks["SpikeSuM_module"]["states"] = self.states
        param_networks["SpikeSuM_module"]["n_memory"] = self.n_memory
        param_networks["selector_module"]["n_memory"] = self.n_memory
        param_networks["SpikeSuM_module"]['batch_size'] = self.batch_size
        param_networks["selector_module"]['batch_size'] = self.batch_size
        param_networks["selector_module"]['input_shape'] = param_networks["SpikeSuM_module"]['EI_neurons']

        # Create Hidden layers (SpikeSuM and selector)
        self.SpikeSuM_module = SpikeSuM.SpikeSuM(param_networks["SpikeSuM_module"])
        self.selector_module = sm.Selector_module(
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
        active_memory_activity = self.selector_module.active_memory_activity
        y = torch.argmax(active_memory_activity, axis=-1)
        one_hot = torch.nn.functional.one_hot(y, num_classes=self.n_memory)
        return one_hot, y

    def initiate_info(self):
        """
        Dic info is a dictionnary saving intersting / to plot data of a simulation
        """
        self.info = {}
        self.info["error"] = torch.zeros(self.batch_size, 1).to(device)
        self.info["estimated_mazes"] = torch.zeros(self.batch_size, 1).to(device)

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
            self.SpikeSuM_module.plot_network()
            if self.n_memory>1:
                self.selector_module.plot_network()

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
        error_array = self.SpikeSuM_module.info['error'][-1]
        plot_times = [1, 500, 1500, 4000, simulation["epochs"] - 1]
        if torch.argmin(error_array) == estimated_active_memory:
            self.time_good_maze += 1
        criteria[b]['count'][estimated_active_memory] += 1
        if maze not in criteria[b]['observed_mazes']:
                maze_n = int(maze.item())
                criteria[b]['maze_{0}'.format(maze_n)] = 1 if 'maze_{0}'.format(maze_n) not in criteria[b] else 1 + criteria[b]['maze_{0}'.format(maze_n)]
                if criteria[b]['maze_{0}'.format(maze_n)] > 50:
                    criteria[b]['observed_mazes'] = torch.cat(
                        (criteria[b]['observed_mazes'], torch.Tensor([simulation["maze"][epoch]]).to(device))).detach()
        if old_memory != estimated_active_memory and epoch > 1:
            shape = self.selector_module.WTA_matrix.shape
            if self.print:
                print(
                    'Simulation N', b + 1,
                    " Epoch", epoch,": Change-point detected. Activating memory : ",
                    estimated_active_memory.item(),file=self.f)
                if maze not in criteria[b]['observed_mazes'] and estimated_active_memory not in criteria[b]['activated_memory']:
                    print("First maze encounter and memory never used", file=self.f)
                elif torch.eq(torch.argmin(error_array[b]), estimated_active_memory):
                    print("Successfully retrieve memory", file=self.f)
                else:
                    print("Wrong memory retrieved", file=self.f)
                
            if estimated_active_memory not in criteria[b]['activated_memory']:
                criteria[b]['activated_memory'] = torch.cat(
                    (criteria[b]['activated_memory'], torch.unsqueeze(
                        estimated_active_memory, 0).float())).detach()
            if criteria[b]['cp_detected'] == False or epoch < 10:
                criteria[b]['time_to_detect_cp'] = torch.cat((criteria[b]['time_to_detect_cp'], torch.Tensor([
                                                             criteria[b]['before_detecting_cp']]).to(device))).detach()

            criteria[b]['cp_detected'] = True
            old_memory = estimated_active_memory
            if criteria[b]['last_change_point'] > 30:
                criteria[b]['wrong_change_point'] += 1
        if self.batch_size < 2 and (epoch in plot_times or epoch % 100 == 0) and self.plot:
            self.plot_network(epoch)

        return old_memory

