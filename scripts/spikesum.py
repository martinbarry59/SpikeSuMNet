import os
import sys
import torch
path = '../'
sys.path.insert(1, '{0}/pkg/'.format(path))

import simulation_utils
import matplotlib.pyplot as plt
import warnings
import tqdm
import pickle
import pickle5 as pickle
import SpikeSuMC_network as SpikeSuMC_network
from run_simulation import run_simulation
import numpy as np
warnings.filterwarnings("ignore")
outputfile = open('logs/logs_single_run.txt', "w")
outputfile.close()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0) # for reproducibility

deter_start = None # determine time without switch at the beginning
number_rooms = 16
n_moves = 2 # number possible directions action K = 2 * n_moves (K number of transition)
n_maze = 4 # number of different mazes
volatility = 0.0005 
epochs = 10
batch_size = 1 #number of simulations. 

params_path = 'params/params_network.pkl'.format(n_moves, volatility)
seeds = []
# IF no file params use set_params in folder params ! 
with open('params/params_network_n_moves_{0}_H_{1}.pkl'.format(n_moves,volatility), 'rb')  as f:
        params= pickle.load(f)

simulations = []
with torch.no_grad():
    # Initilisation all simulation in batch dimension (all sim are run in parallel)
    for i in range(batch_size):
        print('Simulation {0}'.format(i+1))
        seed = round(2**32 * torch.rand(1).item() - 1)
        seeds += [seed]
        print(seed)
        simulations += [simulation_utils.create_simulation(epochs = epochs, number_rooms = number_rooms, volatility = volatility
         , n_moves = n_moves, n_maze = n_maze, seed  = seed, Dirichlet = 0,deter_start = deter_start,symmetric = True)]
        print('change points:')
        for key, value in sorted(simulations[i]['change_points'].items()):
            print((key,value), end =" ")
        print()
    
    # Running simulation
    params['batch_size'] = batch_size
    net = SpikeSuMC_network.SpikeSuMC(params,None)
  
save = 'HD'
net.SpikeSuM_module.tosave = save
criteria,epoch = run_simulation(simulations,net)

# Save and print results
torch.save(criteria,'../results/criteria_move_{2}_H_{0}_batch_{1}'.format(volatility,batch_size,n_moves))
print('number of succes:', round(100 * torch.mean(torch.Tensor([1-criteria[i]['stop'] for i in range(batch_size)])).item() ,2))