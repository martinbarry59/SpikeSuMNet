# +
import os
import sys
import torch
path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.insert(1, '{0}/pkg/'.format(path))
import simulation_utils
import pickle
# import pickle5 as pickle # import pickle if not pickle5
import SpikeSuMC_network as SpikeSuMC_network
from run_simulation import run_simulation
import save_utils

outputfile = open(path+'/scripts/logs/logs_single_run.txt', "w")
outputfile.close()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# -

torch.manual_seed(2) # for reproducibility

deter_start = None # determine time without switch at the beginning
number_rooms = 16
n_moves = 2 # number possible directions action K = 2 * n_moves (K number of transition)
n_maze = 4 # number of different mazes
volatility = 0.002 
epochs = 10000
batch_size = 1 #number of simulations. 

seeds = []
# IF no file params use set_params in folder params ! 
with open(path+'/scripts/params/params_network_n_moves_{0}_H_{1}.pkl'.format(n_moves,volatility), 'rb')  as f:
        params= pickle.load(f)

params['plot'] = True
params['SpikeSuM_module']['plot'] = True

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

for memories in [1,2,3,4]:
    params['n_memory'] = memories
    net = SpikeSuMC_network.SpikeSuMC(params,None)
    save = None # choose 'HD' 'PE' or None
    net.SpikeSuM_module.tosave = save
    criteria,epoch = run_simulation(simulations,net)
    error = net.SpikeSuM_module.info['error']
    that = net.SpikeSuM_module.info['T_hat']
    net.SpikeSuM_module.info.clear() 
    net.SpikeSuM_module.info['error'] = error
    net.SpikeSuM_module.info['T_hat'] = that
    if params['SpikeSuM_module']['random_projection'] == False:
        projection = 'onehot'
    else:
        projection = 'rand'
    save_utils.save(data=net.info, file=path+'/results/SpikeSuM_info_moves_{0}_{1}_modules_{2}'.format(n_moves, projection,memories), type_='pickle')

