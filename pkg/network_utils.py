import torch

def square_EPSC(EPSC_decay, len_epsc, spike_train=None):
    """
    Computation of the exicatory post synaptic current for every neurons

    param EPSC_decay: Estimation of time lapsed since last spike

    len_epsc: The maximum time of an spike induced EPSC

    spike train: index of recently spiking neurons

    return: Neurons EPSC and there update estimation of time since last spike

    """
    EPSC_decay = EPSC_decay - 1 / len_epsc
    EPSC_decay[EPSC_decay < 0] = 0
    EPSC_decay[spike_train > 0] = 1
    EPSC = 0 * EPSC_decay
    EPSC[EPSC_decay > 0] = 1
    return EPSC.detach(), EPSC_decay.detach()


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
