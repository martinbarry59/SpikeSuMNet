CUDA_LAUNCH_BLOCKING=1
import os
import sys
path = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.insert(1, '{0}/../pkg/'.format(path))
import simulation_utils
import plot_utils
import matplotlib.pyplot as plt
import warnings
import pickle
#import pickle5 as pickle
import torch
from SpikeSuMC_network import SpikeSuMC
from run_simulation import run_simulation
import numpy as np
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
infos = []
deter_start = None
number_rooms = 16
n_moves = 2
n_maze = 4
volatility = 0.002
epochs = 10000
batch_size = 100
params_path = path+'/params_network_n_moves_{0}_H_{1}.pkl'.format(n_moves, volatility)
write_file = path + '/logs/log_optimisation_{0}_{1}.txt'.format(n_moves, volatility)


seeds = []
with open(params_path, 'rb') as f:
        params= pickle.load(f)
simulations = []
with torch.no_grad():
    for i in range(batch_size):
        print('Simulation {0}'.format(i+1))
        seed = round(2**32 * torch.rand(1).item() - 1)
        #seed = 79414783
        seeds += [seed]
        print(seed)
        simulations += [simulation_utils.create_simulation(epochs = epochs, number_rooms = number_rooms, volatility = volatility
         , n_moves = n_moves, n_maze = n_maze, seed  = seed, Dirichlet = 0,deter_start = deter_start,symmetric = True)]
        print('change points:')
        for key, value in sorted(simulations[i]['change_points'].items()):
            print((key,value), end =" ")
        print()
min_error = 100
import skopt
import math
from IPython.display import clear_output
# warnings.filterwarnings("ignore")
def optimise_model(param_network, simulation):
    file=open(write_file, "w").close()
    print('---------------------------START OF OPTIMIZATION--------------------------------------------------', file=open(write_file, "a"))
    global min_error
    min_error = epochs 
    SPACE = [
        skopt.space.Real(0.01, 0.1, name='beta', prior='uniform'),
        skopt.space.Real(0.05, 1, name='r', prior='uniform'),
        skopt.space.Real(0.00001, 0.001, name='lr_msp', prior='uniform'),
        skopt.space.Real(0.01, .2, name='max_commitment', prior='uniform'),
#         skopt.space.Real(0.01, 0.1, name='wta_speed', prior='uniform')
    ]

    @skopt.utils.use_named_args(SPACE)
    def objective(**params):
        
        global min_error
        all_params = {**params}
        print(all_params, file=open(write_file, "a") )
        param_network['plot'] = False
        param_network['SpikeSuM_module']['plot'] = False
        param_network['print'] = True
        param_network['batch_size'] = batch_size
        param_network['n_memory'] = 5
        param_network['SpikeSuM_module']['eta1'] = 1e-05
        param_network['SpikeSuM_module']['eta2'] = 0.005
        param_network['SpikeSuM_module']['theta'] = 0.45
        param_network['selector_module']['beta'] = all_params['beta']
        param_network['selector_module']['r'] = all_params['r']
        param_network['selector_module']['lr_msp'] = all_params['lr_msp']
        param_network['selector_module']['max_commitment'] = all_params['max_commitment']
        param_network['SpikeSuM_module']['W_init'] = 60
        # param_network['SpikeSuM_module']['W_init'] = 45
#         param_network['selector_module']['wta_speed'] = all_params['wta_speed']
        net = SpikeSuMC(param_network,None)
        criteria, epoch = run_simulation(simulations,net)
        error = torch.mean(net.info["error"]).item() + (epochs - epoch - 1)
        prefix ='results_opti'
        if math.isnan(error):
            error = 10000
        if error < min_error:

#             clear_output(wait=True)
            print('Min error : {0} for {1} epochs done \n'.format(error,epoch))
            print('Min error : {0} for {1} epochs done \n'.format(error,epoch), file=open(write_file, "a"))
            
            min_error = error
            plot_utils.plotly_plot([net.info['error'][0].cpu()],['time step','Error(t)'],prefix+'_Error')
            values = [1-criterium['stop'] for criterium in criteria]
            print('mean success: ', round(np.mean(values)*100,1),'$\pm$',round(np.std(values)/np.sqrt(len(criteria))*100,1), file=open(write_file, "a"))
            values = np.array([(len(criterium['observed_mazes'])== len(criterium['activated_memory'])) for criterium in criteria ])
            print('mean memory usage: ',round(np.mean(values)*100,1),'$\pm$',round(np.std(values)/np.sqrt(len(criteria))*100,1), file=open(write_file, "a"))
            values = np.array([(len(criterium['observed_mazes'])== len(criterium['activated_memory']))* (1-criterium['stop']) for criterium in criteria])
            print('mean  total success: ',round(np.mean(values)*100,1),'$\pm$',round(np.std(values)/np.sqrt(len(criteria))*100,1), file=open(write_file, "a"))
            values = np.array([criterium['before_detecting_cp']  for criterium in criteria if 1*criterium['stop']==0])
            print('mean detection time: ',round(np.mean(values),1),'$\pm$',round(np.std(values)/np.sqrt(len(criteria)),1), file=open(write_file, "a"))
        return error
    try:
        filename = path+'/results_opti.pkl'
        results = skopt.load(filename)
        results = skopt.gp_minimize(objective, SPACE, n_jobs = -1, noise=1e-10, x0 = results.x,y0= results.fun)
    except:    
        results = skopt.gp_minimize(objective, SPACE, n_calls = 500, n_random_starts = 100)
    
    skopt.dump(results,path+'/results_opti.pkl',store_objective=False)

    print(':min error of:', results.fun,' for values:', results.x, file=open(write_file, "a"))
    print('---------------------------END OF OPTIMIZATION--------------------------------------------------', file=open(write_file, "a"))


optimise_model(params, simulations)