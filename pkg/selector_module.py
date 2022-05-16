import torch
import network_utils
import plot_utils
import matplotlib.pyplot as plt
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not os.path.exists("logs"):
    os.mkdir("logs")


class Selector_module(object):
    def __init__(
        self,
        params,
    ):
        """
        Initialize Disinhibitory_module class
        params: Dictionnary including all network params. More details in
            results/Set_mFasNet_params.ipynb
        """
        # Network properties
        self.n_memory = params["n_memory"]
        self.msp_neurons = params["msp_neurons"]
        self.input_shape = params["input_shape"]
        self.batch_size = params['batch_size']
        self.plot = params["plot"]

        # Neuron model
        self.len_epsc = params["l"]
        self.decay = 0.9
        self.tau = params["tau"]
        self.wta_speed = params["a2"]
        self.lr_msp = params["lr_msp"]

        # WTA parameters
        self.beta = torch.FloatTensor([params["beta"]]).to(device)
        self.r = params["a1"]
        self.self_inhib = params["a3"]
        self.module_inhib = params["a4"]
        self.alpha_p = params["alpha_p"]
        
        # Layer initialisation
        self.init_layer()

        # Weight initialisation
        self.FF_scaling = self.input_shape
        (
            self.FF_msp_weights,
            self.FB_msp_weights,
            self.WTA_matrix,
            self.commitment_matrix,
        ) = self.initiate_msp_weights()
        self.commitment_matrix_fix = self.commitment_matrix.clone()
        self.FF_msp_weights_fix = self.FF_msp_weights.clone()
        self.max_commitment = params["Wmax"]
        
        self.initiate_info()

    def initiate_msp_weights(self):
        """
        Initialise Weights between EI mismatch to memory selector, It also implements between the neurons in the WTA network
        """
        bloc_FF_msp_weights = torch.ones(
            (self.batch_size, self.input_shape, self.msp_neurons))
        bloc_shape = bloc_FF_msp_weights.shape
        bloc_WTA_matrix = torch.ones(
            (self.batch_size, self.msp_neurons, self.msp_neurons))
        FF_msp_weights = torch.zeros(
            (self.batch_size,
             self.n_memory *
             self.input_shape,
             self.n_memory *
             self.msp_neurons))
        WTA_matrix = torch.zeros(
            (self.batch_size,
             self.n_memory *
             self.msp_neurons,
             self.n_memory *
             self.msp_neurons))
        idxx = 0
        idxy = 0
        for memory in range(self.n_memory):
            bloc_shape = bloc_FF_msp_weights[0].shape
            idxx = bloc_shape[0] * memory
            idxy = bloc_shape[1] * memory
            FF_msp_weights[:, idxx:idxx + bloc_shape[0],
                           idxy:idxy + bloc_shape[1]] = bloc_FF_msp_weights
            bloc_shape = bloc_WTA_matrix[0].shape
            idxx = bloc_shape[0] * memory
            idxy = bloc_shape[1] * memory
            WTA_matrix[:, idxx:idxx +
                       bloc_shape[0], idxy:idxy +
                       bloc_shape[1]] = bloc_WTA_matrix

        FB_msp_weights = 1 - FF_msp_weights.clone()
        commitment_matrix = WTA_matrix.clone() / self.msp_neurons
        WTA_matrix = (1 - WTA_matrix) .clone() / self.msp_neurons
        return FF_msp_weights.to(device), FB_msp_weights.to(
            device), WTA_matrix.to(device), commitment_matrix.to(device)

    def initiate_info(self):
        """
        Dic info is a dictionnary saving intersting / to plot data of a simulation
        """
        self.info = {}
        self.info["msp_input"] = []
        self.info["eta"] = []
        self.info["msp"] = []
        self.info["msp_spikes"] = []
        self.info["inhib_spikes"] = []
        self.info["WTA"] = []

    def init_layer(self):
        """
        Initialisation to 0 of all neurons potential etc.. of the network
        """
        self.msp_input_potential = torch.zeros(
            self.batch_size, self.n_memory * self.msp_neurons).to(device)
        self.inhibitory_eta_msp = torch.zeros_like(
            self.msp_input_potential).to(device)
        self.refractoriness = torch.zeros_like(
            self.msp_input_potential).to(device)
        self.inhibitory_msp_input_potential = torch.zeros_like(
            self.msp_input_potential).to(device)
        self.msp_spikes = torch.zeros(
            self.batch_size,
            self.n_memory *
            self.msp_neurons).to(device)
        self.inhibitory_msp_spikes = torch.zeros(
            self.batch_size, self.n_memory * self.msp_neurons).to(device)
        self.EPSC_msp_decay = torch.zeros(
            self.batch_size,
            self.n_memory *
            self.msp_neurons).to(device)
        self.EPSC_inhibitory_msp_decay = torch.zeros(
            self.batch_size, self.n_memory * self.msp_neurons).to(device)
        self.active_memory_activity = torch.zeros(
            self.batch_size, self.n_memory).to(device)
        self.EPSC_msp_decay = torch.zeros(
            self.batch_size,
            self.n_memory *
            self.msp_neurons).to(device)
        self.wta_output = torch.zeros(
            self.batch_size,
            self.n_memory *
            self.msp_neurons).to(device)
        self.filtered_inhibition = torch.zeros(
            self.batch_size,
            self.n_memory).to(device)
    def phi(self, x):
        """
        error neuron activation function

        param x: neuron membrane potential

        return $$f(x)$$
        """
        return (x > 0).float() * torch.tanh(x)

    def update_pot(self, h, I):
        """
        Layer update

        param h: input potential

        param I: input current

        return: Integrated potential
        """
        h += 1 / self.tau * (-h + I)
        return h

    def wta_update_module(self, x, msp_spikes, wta_input, meanput):
        """
        Runs one step of the WTA dynamic on the module.

        param x: neuron self input potential

        param msp_spikes: Inhibition coming from self memory (More inhibition implies more error in SpikeSumNet)

        param wta_input: Inhibition coming from other memories (More inhibition implies other memory active)

        return: Updated input potential
        """

        step = self.wta_speed * (
            - x
            - self.self_inhib *
            torch.einsum("bkj,bj->bk", self.commitment_matrix, msp_spikes)
            - self.module_inhib * wta_input
            + meanput.repeat(wta_input.shape[-1],1).T  
        )

        self.input = self.self_inhib * \
            torch.einsum("bkj,bj->bk", self.commitment_matrix, msp_spikes).detach().clone()

        x += step

        return x.detach()

    def update_layer(self, input_potential, EPSC_decay, refractoriness):
        """
        Full update of the error layer

        Param input_potential: Input potential receive by the neurons

        Param EPSC_decay: Estimation of time since last spike of error neurons

        refractoriness: refractoriness value (inhibition due to recent spike)

        returns: new spikes index, active EPSC, estimation of time since last spike, refractoriness
        """

        spikes = 0 * input_potential
        membrane_potential = input_potential - refractoriness
      
        idx = self.phi(membrane_potential) > torch.Tensor(
            self.batch_size, self.msp_neurons * self.n_memory).uniform_().to(device)
        spikes[idx] = 1.0
        membrane_potential[idx] = 0
        EPSC, EPSC_decay = network_utils.square_EPSC(EPSC_decay, self.len_epsc, spikes)
        refractoriness *= self.decay
        refractoriness[idx] = 1
        return spikes, EPSC, EPSC_decay, refractoriness

    def clear_spike_train(self):
        """
        allow suppressing all saved spike times
        """
        self.info["msp_spikes"] = []
        self.info["inhib_spikes"] = []

    def forward(self, input_, commitement_modulation, learning=True):
        """
        Disinhibitory module step

        param input_: Spiking activity coming from SpikeSumNet

        param commitement_modulation: Modulation information receive from SpikeSumNet

        param learning: switch learning off (debugging only)
        """

        msp_input = self.r * \
            (torch.einsum("bjk,bj->bk", self.FF_msp_weights, input_ - self.beta)).detach()
        
        self.meanput =  20 * torch.mean(input_, dim=1).detach() 
        
        wta_input = self.wta_output.detach().clone()
        self.msp_input_potential = self.update_pot(
            self.msp_input_potential,
            msp_input,
        )
        (
            self.msp_spikes,
            EPSC_msp,
            self.EPSC_msp_decay,
            self.refractoriness,
        ) = self.update_layer(
            self.msp_input_potential, self.EPSC_msp_decay, self.refractoriness
        )
        self.inhibitory_msp_input_potential = self.wta_update_module(
            self.inhibitory_msp_input_potential,
            EPSC_msp,
            wta_input,
            self.meanput 
        )
        (
            self.inhibitory_msp_spikes,
            EPSC_inhibitory_msp,
            self.EPSC_inhibitory_msp_decay,
            self.inhibitory_eta_msp,
        ) = self.update_layer(
            self.inhibitory_msp_input_potential,
            self.EPSC_inhibitory_msp_decay,
            self.inhibitory_eta_msp,
        )
        self.active_memory_activity += (1 / self.tau * (-self.active_memory_activity + torch.sum(
            self.inhibitory_msp_spikes.reshape(self.batch_size, self.n_memory, -1), axis=2))).detach()

        if learning:

            commitement_modulation = commitement_modulation.repeat_interleave(self.msp_neurons, 1).detach()
            self.commitment_matrix += (self.lr_msp *
                                       torch.einsum("bj,bk->bjk", self.msp_spikes, commitement_modulation *
                                                    self.inhibitory_msp_spikes) -
                                       (self.commitment_matrix -
                                        self.commitment_matrix_fix) *
                                       self.alpha_p *
                                       0).detach()
            self.commitment_matrix *= 1 * (self.commitment_matrix_fix > 0)
            self.commitment_matrix[self.commitment_matrix <
                                   self.commitment_matrix_fix] = self.commitment_matrix_fix[self.commitment_matrix < self.commitment_matrix_fix].detach()
            self.commitment_matrix[self.commitment_matrix > self.max_commitment] = self.max_commitment# move 

        self.memory_output = torch.einsum(
            "bjk,bk->bj",
            self.FB_msp_weights,
            EPSC_inhibitory_msp).unsqueeze(1).repeat(1, 2, 1).detach()
        self.wta_output = torch.einsum(
            "bjk,bk->bj",
            self.WTA_matrix,
            EPSC_inhibitory_msp).detach()
        self.info["msp_spikes"] += [self.msp_spikes]
        self.info["inhib_spikes"] += [self.inhibitory_msp_spikes]
    def plot_network(self):
        """
        Plotting few properties of the network; Spike train of the two neuronal population and the commitment matrix.
        """
        if self.plot:
            print('---------Selector Network---------')
            print('---First layer spikes----')
            plot_utils.plot_spike_train(
                torch.transpose(
                    torch.cat(
                        self.info["msp_spikes"]).view(
                        -1,
                        self.msp_neurons * self.n_memory),
                    0,
                    1).cpu().detach(),
                self.msp_neurons,
                1,
                title='memory selector spike train layer 1',
                directory="{0}_spikes_plot".format(0),
            )
            print('---Second layer spikes----')
            plot_utils.plot_spike_train(
                torch.transpose(
                    torch.cat(
                        self.info["inhib_spikes"]).view(
                        -1,
                        self.msp_neurons * self.n_memory),
                    0,
                    1).cpu().detach(),
                self.msp_neurons,
                1,
                title='memory selector spike train layer 2',
                directory="{0}_spikes_plot".format(0),
            )
            print('---Commitment matrix----')
            for context in range(self.commitment_matrix.shape[0]):
                plt.imshow(self.commitment_matrix.cpu()[context].detach())
                plt.colorbar()
                plt.show()


