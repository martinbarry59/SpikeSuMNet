# coding: utf-8
"""EI-mismatch network, prediction and error estimation"""
from scipy.sparse import rand
import matplotlib.pyplot as plt
import torch
import network_utils
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SpikeSuM(object):
    """Network Class"""

    def __init__(
        self,
        params,
    ):
        """
        Initialize SpikeSuM class
        params: Dictionnary including all network params. More details in
            results/Set_SpikeSuM-M_params.ipynb
        """
        # Environment properties
        self.number_rooms = params["number_rooms"]
        self.states = params["states"]

        # Network properties
        self.n_memory = params["n_memory"]
        self.EI_neurons = params["EI_neurons"]
        self.input_neurons = params["input_neurons"]
        self.batch_size = params['batch_size']
        self.sparsity = 0.1 ## Hard coded because never change.
        self.plot = params["plot"]
        self.modulation = params["modulation"]
        
        self.rooms_encoding = torch.zeros(
            (self.batch_size, self.number_rooms, self.input_neurons)).to(device)

        # Neuron model
        self.tau = params["tau"]
        self.eta1 = params["eta1"]
        self.eta2 = params["eta2"]
        self.theta = torch.Tensor([params["theta"]]).to(device)
        self.N = params["N"]
        self.Poisson_rate = params["Poisson_rate"]
        self.tau = params["tau"]
        self.len_epsc = params["l"]
        self.decay = 0.9 ## Hard coded because never change.
        self.sign = torch.Tensor([[1], [-1]]).to(device)
        self.scale = (self.EI_neurons / 128)
        self.FB_inhib = params["FB_inhib"]
        self.decay_factor =  (1 - torch.exp(torch.Tensor([-1 / self.tau]))).to(device)
        
        # Initialiste the layer population
        self.initiate_layer()
       

        for i in range(self.number_rooms):
            self.rooms_encoding[:, i, self.states[i]] = 1

        # Weights initialisation
        self.W = params["W"]
        if self.W is None:
            # Scaling is important for

            self.W =  torch.rand(self.batch_size,2, self.input_neurons, self.n_memory * self.EI_neurons).to(device) / (params['W_init'] * self.scale)
        self.observation_weights, self.readout_weights = self.initiate_feedback(
            params["random_projection"])
        if not params["random_projection"]:
            self.A0 = (
                .2
                * self.EI_neurons
                / self.number_rooms
                * self.phi(torch.Tensor([1]).to(device))
                * (self.input_neurons / self.number_rooms)
            )  # Expectation of maximum neuron active at the same time
        else:
            self.A0 = (
                sparsity
                * .2
                * self.EI_neurons
                * self.phi(torch.log(torch.cosh(1)))
                * (self.input_neurons / self.number_rooms)
            )

        self.initiate_info()
    def initiate_layer(self):
        """
        Set all main layer dependencies
        """
         # Layer initialisation
        self.h = torch.zeros(
            (self.batch_size, 2, self.n_memory * self.EI_neurons)).to(device)
        self.u = torch.zeros(
            (self.batch_size, 2, self.n_memory * self.EI_neurons)).to(device)
        self.refractoriness = torch.zeros_like(self.h)
        self.EPSC_EI_decay = torch.zeros(
            (self.batch_size, 2, self.n_memory * self.EI_neurons)).to(device)
        self.filtered_activity = torch.zeros(
            (self.batch_size, 2, self.n_memory * self.EI_neurons)).to(device)
        self.filtered_theta = torch.zeros(
            self.batch_size, self.n_memory).to(device)
        self.filtered_EPSC = torch.zeros(
            (self.batch_size, 2, self.input_neurons)).to(device)
        self.network_activity = torch.zeros(self.n_memory)
        self.output = torch.zeros(
            self.batch_size,
            2 * self.EI_neurons).to(device)
    def initiate_info(self):
        """
        Instantiate all info to save from a simulation
        """
        self.info = {}
        self.info["Activity"] = []
        self.info["Activity_full"] = [[] for _ in range(self.n_memory)]
        self.info["Activity_P1"] = []
        self.info["Activity_P2"] = []
        self.info["error"] = []
        self.info["weights"] = []
        self.info["spikes"] = []
        self.info["EPSC"] = []
        self.info["T1"] = []
        self.info["T2"] = []
        self.info["T_hat"] = torch.zeros(
            (self.number_rooms, self.number_rooms))
        self.info["Learning_rate"] = []
        self.info["readout_weights"] = self.readout_weights
        self.info["EI_spikes"] = []
        self.info['absolute error'] = []
        self.info['effective update'] = []
        self.info['prediction error'] = []
        self.info['effective update_pos'] = []
        self.info['prediction error_pos'] = []
        self.info['effective update_neg'] = []
        self.info['prediction error_neg'] = []
    def initiate_feedback(self, random_projection=False):
        """
        Observation weight initialisation.

        param random_projection: True/False; declares whether room observation is one hot encoded or randomly projected

        returns: Observation weights (Projection onto the error layer); readout weights (allow memory wise decoding of Prediction weights)
        """

        readout_weights = []
        if not random_projection:
            observation_weights = torch.zeros(
                (self.batch_size, 2, self.input_neurons, self.n_memory * self.EI_neurons)).to(device)
            diff = self.input_neurons / self.EI_neurons
            for memory in range(self.n_memory):

                for room in self.rooms_encoding[0]:

                    nk = torch.sum(room) * self.Poisson_rate * self.len_epsc

                    idx = torch.nonzero(room)[0][0]
                    length = int(torch.sum(room))
                    observation_weights[:, :, idx: idx + length, int(1.0 * idx / diff) + memory * self.EI_neurons: int(
                        1.0 * (idx + length) / diff + memory * self.EI_neurons), ] = 2.0 / nk
        if random_projection:
            sparsify = rand(
                self.input_neurons,
                self.n_memory * self.EI_neurons,
                density=self.sparsity,
                format="csr",
            )
            sparsify.data[:] = 1
            observation_weights = torch.rand(
                self.batch_size,
                2,
                self.input_neurons,
                self.n_memory *
                self.EI_neurons)
            observation_weights[0] *= sparsify.toarray().clone()
            sparsify = rand(
                self.input_neurons,
                self.n_memory * self.EI_neurons,
                density=self.sparsity,
                format="csr",
            )
            sparsify.data[:] = 1
            observation_weights[1] *= sparsify.toarray().clone()
        for i in range(2):
            self.R = self.rooms_encoding.clone()
            pop_weights = []
            for memory in range(self.n_memory):
                P = observation_weights[:, i, :, memory *
                                        self.EI_neurons:memory *
                                        self.EI_neurons +
                                        self.EI_neurons].clone()
                readout = torch.pinverse(self.R @ P)
                pop_weights += [torch.unsqueeze(readout, 1)]
            readout_weights += [
                torch.unsqueeze(
                    torch.cat(
                        pop_weights,
                        dim=1),
                    1)]
        readout_weights = torch.cat(readout_weights, dim=1).to(device)
        return observation_weights.to(device), readout_weights.to(device)

    def estimate_T(self):
        """
        Decode prediction weights

        return: The estimated transition matrix $$T=\frac{1}{2}(T_1+T_2)$$
        """
        shape = self.W.shape
        W_reshaped = self.W.view(
            shape[0],
            shape[1],
            shape[2],
            self.n_memory,
            int(shape[3]/self.n_memory)).permute(
            0,
            1,
            3,
            2,
            4)
        Transition_hidden_space = torch.einsum(
            "bijkl,bijlm->bijkm", W_reshaped, self.readout_weights)
        Transition_one_hot_space = torch.einsum(
            "bijkl,bmk->bijlm", Transition_hidden_space, self.R)
        T = torch.mean(Transition_one_hot_space, dim=1)  # T1 + T2 average
        T = torch.transpose(T, 2, 3)
        T = torch.nn.functional.normalize(T, p = 1.0, dim = 3)
        return T

    def clear_spike_train(self):
        """
        Delete the spike train we keep in memory
        """
        self.info["EI_spikes"] = []

    def phi(self, x):
        """
        error neuron activation function

        param x: neuron membrane potential

        return $$f(x)$$
        """
        return (x > 0).float() * torch.tanh(x)

    def third_factor(self, x):
        """
        Modulatory learning signal

        param x: network activity

        param self.modulation: define whether we use constant learning rate, single of full modulation

        param self.theta: Surprise level, if $$x>\theta$$ the agent is considered in a Surprised state

        return: Surprise modulatory signal signal
        """
        if self.modulation == 'full':
            return (self.eta1 * torch.tanh((x)) + self.eta2 * \
                    torch.tanh(x) * (x >= self.theta).float())  * (x > 0).float()
        elif self.modulation == 'single':
            return self.eta1 * torch.tanh((x)) * (x > 0).float()
        elif self.modulation == 'step':
            return (self.eta1 +
                    self.eta2 *
                    (x >= self.theta).float() *
                    (x > 0).float())
        elif self.modulation == 'none':
            return self.eta1 * (x > 0)

    def update_pot(self, h, I):
        """
        Layer update

        param h: input potential

        param I: input current

        return: Integrated potential
        """

        h += 1 / self.tau * (-h + I)
        return h

    def update_layer(self, I, EPSC_decay):
        """
        Full update of the error layer

        Param I: Input current receive from both prediction and observation

        Param EPSC_decay: Estimation of time since last spike of error neurons
        
        return: Spikes, EPSC,  time since last spike (in the form of decaying EPSC)
        """

        spikes = 0 * self.h
        self.h = self.update_pot(self.h, I)
        self.u = self.h.clone() - self.refractoriness
        self.ratio = torch.mean(self.u[self.u > 0])

        idx = self.phi(self.u) >  torch.Tensor(self.batch_size,
                                                    2, self.n_memory * self.EI_neurons).uniform_().to(device)
        spikes[idx] = 1.0
        EPSC, EPSC_decay = network_utils.square_EPSC(EPSC_decay, self.len_epsc, spikes)
        self.refractoriness *= self.decay
        self.refractoriness[idx] = 1
        return spikes.detach(), EPSC.detach(), EPSC_decay.detach()

    def save_prediction(self, current_T_matrix):
        """
        Saving the Matrix transition estimation error

        param current_T_matrix: True maze transition matrix to be estimaed

        Return None
        """

        T_hat = self.estimate_T()

        self.info["error"] += [
            torch.mean(
                (current_T_matrix - T_hat)**2,
                dim=(
                    2,
                    3)).clone().detach()]
        self.info["T_hat"] = T_hat.clone().detach()

    def forward(
        self, EPSC_buffer, EPSC_observation, module_inhib, learning=False
    ):
        """
        SpikeSumNet step

        param EPSC_buffer: The active neurons in the buffer population

        param EPSC_observation: The active neurons in the observaiton  population

        module_inhib: Feedback inhibition coming from the dishinibitory modules (if multiple memories)

        param learning: switch learning off (debugging only)

        return: Average weight update and commitment modulation for disinhibitory neurons
        """

        I = (self.sign * (torch.einsum("bijk,bij->bik",
                                       self.W,
                                       EPSC_buffer) - torch.einsum("bijk,bij->bik",
                                                                   self.observation_weights,
                                                                   EPSC_observation)).detach())
        (
            self.EI_spikes,
            EPSC_EI,
            self.EPSC_EI_decay,
        ) = self.update_layer(I, self.EPSC_EI_decay)
        self.filtered_activity = self.filtered_activity + (1.0 / self.N) * (
            - self.filtered_activity + (self.EI_spikes  - self.FB_inhib * module_inhib) / self.A0 
        ).detach()
        memory_activity = torch.sum(
            self.filtered_activity, dim=1).reshape(
            self.batch_size, self.n_memory, -1)
        self.network_activity = torch.sum(memory_activity, dim=2).detach()
        
        self.info["Activity_P1"] += [torch.sum(self.filtered_activity[0,0])]
        self.info["Activity_P2"] += [torch.sum(self.filtered_activity[0,1])]
        self.info["Activity"] += [self.network_activity.detach().clone()]
        
        third = self.third_factor(self.network_activity).detach()
        commitement_modulation = (third > 0 ) * (1 - 2 * (third > self.third_factor(self.theta)))
        prediction_modulation = third.detach()

        third = torch.unsqueeze(
            third.repeat_interleave(
                self.EI_neurons,
                1),
            dim=1).repeat_interleave(
            2,
            dim=1).detach()

        if learning:
            self.filtered_EPSC = self.decay_factor * self.filtered_EPSC + EPSC_buffer * (1 - self.decay_factor)
            deltaW = (
                torch.einsum(
                    "bij,bik->bijk",
                    self.filtered_EPSC,
                    third *
                    self.h)).detach()
            self.W -= torch.einsum("ij,bikl->bikl", self.sign, deltaW).detach()
            self.W[self.W < 0] = 0
            self.W = self.W.detach()
        
        self.output = torch.sum(EPSC_EI, dim = 1).detach() / 2
        self.input = self.network_activity .detach()
        self.save_drives(third, deltaW)
        
        return prediction_modulation, commitement_modulation
    
    def save_drives(self,third,deltaW):
    ## Prediction error Drive 
        if self.batch_size == 1:
            if self.tosave == 'PE':
                self.info['effective update'] += [torch.mean((third * torch.abs(self.h)).detach().clone(),axis=1)]
                self.info['prediction error'] += [torch.mean(torch.abs(self.h).detach().clone(),axis=1)]
                self.info['effective update_pos'] += [(third[0,0] * torch.abs(self.h[0,0])).detach().clone()]
                self.info['prediction error_pos'] += [torch.abs(self.h[0,0]).detach().clone()]
                self.info['effective update_neg'] += [(third[0,1] * torch.abs(self.h[0,1])).detach().clone()]
                self.info['prediction error_neg'] += [torch.abs(self.h[0,1]).detach().clone()]
            ## Hebbian Drive
            if self.tosave == 'HD':
                Hebbian_drive = torch.einsum(
                           "bij,bik->bijk",
                           self.filtered_EPSC,
                           self.h)
                self.info['effective update'] += [torch.abs(deltaW[Hebbian_drive != 0].detach().clone()).cpu()]
                self.info['prediction error'] += [torch.abs(Hebbian_drive[Hebbian_drive != 0]).detach().clone().cpu()]
                self.info['effective update_pos'] += [torch.abs(deltaW[0,0][Hebbian_drive[0,0] != 0]).detach().clone().cpu()]
                self.info['prediction error_pos'] += [torch.abs(Hebbian_drive[0,0][Hebbian_drive[0,0] != 0]).detach().clone().cpu()]
                self.info['effective update_neg'] += [torch.abs(deltaW[0,1][Hebbian_drive[0,1] != 0]).detach().clone().cpu()]
                self.info['prediction error_neg'] += [torch.abs(Hebbian_drive[0,1][Hebbian_drive[0,1] != 0]).detach().clone().cpu()]
                
    def plot_network(self):
        """
        Plotting few properties of the network; the estimated transition matrix error, the activity  and finally the estimated transition matrix.
        This is shown for every memories. The transition matrix shows the estimated transition for every memory.
        """
        if self.plot:
            print("----- Plot SpikeSuM Module:-----")

            for memory in range(self.n_memory):
                plt.plot(torch.cat(
                    self.info['error']).view(-1, self.n_memory)[:, memory].cpu().detach())
            plt.title('Estimated transition error')
            plt.show()
            for memory in range(self.n_memory):
                x = torch.mean((torch.reshape(torch.cat(
                    self.info['Activity']).view(-1, self.n_memory)[:, memory], (-1, 100))), dim=1)
                plt.plot((x * (x > 0)).cpu().detach())
            plt.title('memory Activity')
            plt.show()
            T = self.info["T_hat"]
            shape = T.shape
            T = T.reshape(-1, T.shape[-1])
            T = torch.nn.functional.normalize(T, p = 1.0, dim = 1)
            plt.imshow(T.cpu().detach().T, aspect=1)
            plt.colorbar()
            plt.show()
