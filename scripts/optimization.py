import os
import sys
import torch
path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.insert(1, '{0}/pkg/'.format(path))
import simulation_utils
import plot_utils
import warnings
import pickle
import SpikeSuMC_network as SpikeSuMC_network
from run_simulation import run_simulation
warnings.filterwarnings("ignore")
outputfile = open(path +'/logs/logs_single_run.txt', "w")
outputfile.close()
import Bayesian_change_detection as bcd
import utils_other_alg


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

min_error = 100
import skopt
import math
from IPython.display import clear_output
# warnings.filterwarnings("ignore")
def optimise_model(param_network, simulations, model):
    """
    Optimize the model parameters using Bayesian optimization.

    Args:
        param_network (dict): Dictionary containing the network parameters.
        simulations (list): List of simulation data, we run multiple simulation at a time to average errors for better param optimizaiton.
        model (str): Name of the model to optimize.

    Returns:
        None
    """
    moves = simulations[0]['n_moves']
    rooms = simulations[0]['number_rooms']
    write_file = path +'/logs/log_optimisation_{0}.txt'.format(model)
    print('---------------------------START OF OPTIMIZATION--------------------------------------------------', file=open(write_file, "w"))
    global min_error
    min_error = epochs
    if model == 'SNN' or 'SNNrand':
        SPACE = [
           skopt.space.Real(1e-8, 1e-5, name='eta1', prior='uniform'),
           skopt.space.Real(1e-5, 1e-2, name='eta2', prior='uniform'),
           skopt.space.Real(0.1, 2., name='theta', prior='uniform')]
    if model == 'BOCPA' or model == 'BOCPA-D':
        SPACE = [
            skopt.space.Real(1e-3, 1, name='S', prior='uniform'),
            skopt.space.Real(1e-3, 1, name='lambda')
           ]
    if model == 'VarSmile':
        SPACE = [
         skopt.space.Real(1e-2, 10, name='m', prior='uniform'),
         skopt.space.Real(1e-1, 10, name='epsilon', prior='uniform')]
    if model == 'SNNnm':
        SPACE = [
         skopt.space.Real(1e-4, 1e-1, name='eta', prior='uniform')]
    if model == 'SNNsm':
        SPACE = [
         skopt.space.Real(1e-4, 1e-1, name='eta', prior='uniform')]

    @skopt.utils.use_named_args(SPACE)
    def objective(**params):
        
        global min_error
        all_params = {**params}
        print(all_params, file=open(write_file, "a") )
        
        param_network['plot'] = False
        param_network['SpikeSuM_module']['plot'] = False
        param_network['print'] = False
        param_network['batch_size'] = batch_size
        param_network["SpikeSuM_module"]['number_rooms'] = rooms
        param_network['n_memory'] = 1
        if 'SNN' in model:
            if model == 'SNN':
                param_network['SpikeSuM_module']['eta1'] = all_params['eta1']
                param_network['SpikeSuM_module']['eta2'] =all_params['eta2']
                param_network['SpikeSuM_module']['theta'] = all_params['theta']
                param_network['SpikeSuM_module']['random_projection'] = False

            if model == 'SNNrand':
                param_network['SpikeSuM_module']['eta1'] = all_params['eta1']
                param_network['SpikeSuM_module']['eta2'] =all_params['eta2']
                param_network['SpikeSuM_module']['theta'] = all_params['theta']
                param_network['SpikeSuM_module']['random_projection'] = True

            if model == 'SNNsm':
                param_network['SpikeSuM_module']['eta1'] = all_params['eta']
                param_network['SpikeSuM_module']['modulation'] = 'single' 
            if model == 'SNNnm':
                param_network['SpikeSuM_module']['eta1'] = all_params['eta']
                param_network['SpikeSuM_module']['modulation'] = 'none' 
            param_network['SpikeSuM_module']['N'] = 50

            net = SpikeSuMC_network.SpikeSuMC(param_network,None)
            net.SpikeSuM_module.tosave = None
            _, _ = run_simulation(simulations,net)
            error = torch.mean(net.info["error"]).item()
        if model == 'BOCPA':
            dic_evolution, error, _ = bcd.Bayesian_change_point(epochs = simulations[0]['epochs'], n_room = simulations[0]['number_rooms'], rmax = 5000, s = all_params['S'], pc = all_params['lambda'], simulation = simulations[0])
        if model == 'VarSmile':
            dic_evolution, error = utils_other_alg.VarSmile(simulations[0]['epochs'], all_params['m'], all_params['epsilon'], simulations[0]['number_rooms'], simulations[0])
                                                            
        prefix ='optimization_{0}_moves_{1}_rooms_{2}'.format(model, moves, rooms)

        if math.isnan(error):
            error = 100
        if error < min_error:
            
            clear_output(wait=True)
            print(model,' moves: ',moves,' rooms:',rooms,' params:',all_params)
            print('Min error : {0}\n'.format(error))
            print('Min error : {0} \n'.format(error), file=open(write_file, "a"))
            
            min_error = error
#             utils.save(net.info, file = prefix, type_= 'data')
            if 'SNN' in model:
                error_fig = net.info['error'][0].cpu()
            else:
                error_fig = dic_evolution['error']
            plot_utils.plotly_plot([error_fig],['time step','Error(t)'],prefix+'_Error',plot = False)
            
        return error
    try:
        
        filename =path+'/results/optimisation/'+'results_optimization_{0}_moves_{1}_rooms_{2}.pkl'.format(model, moves, rooms )
        results = skopt.load(filename)
        results = skopt.gp_minimize(objective, SPACE, n_jobs = -1, noise = 1e-10, x0 = results.x,y0 = results.fun)
    except:
        results = skopt.gp_minimize(objective, SPACE,n_initial_points = 50,n_calls = 200)
    
    skopt.dump(results,path+'/results/optimisation/'+'results_optimization_{0}_moves_{1}_rooms_{2}.pkl'.format(model, moves, rooms ),store_objective=False)

    print(':min error of:', results.fun,' for values:', results.x, file=open(write_file, "a"))
    print('---------------------------END OF OPTIMIZATION--------------------------------------------------', file=open(write_file, "a"))
for model in ['VarSmile']:
    
     for number_rooms in [16,32]:
        for n_moves in [4]:
    
    
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            torch.manual_seed(0)

            infos = []
            deter_start = None
            number_rooms = number_rooms
            n_moves = n_moves
            n_maze = 4
            volatility = 0.002
            epochs = 5000
            batch_size = 1
            params_path = path+'/scripts/params/params_network.pkl'.format(n_moves, volatility)
            seeds = []
            with open(params_path, 'rb') as f:
                    params= pickle.load(f)
            simulations = []
            with torch.no_grad():
                for i in range(batch_size):
                    print('Simulation {0}'.format(i+1))
                    seed = 79414783
                    seeds += [seed]
                    simulations += [simulation_utils.create_simulation(epochs = epochs, number_rooms = number_rooms, volatility = volatility
                     , n_moves = n_moves, n_maze = n_maze, seed  = seed, Dirichlet = 0,deter_start = deter_start,symmetric = True)]
                    print('change points:')
                    for key, value in sorted(simulations[i]['change_points'].items()):
                        print((key,value), end =" ")
                    print()
            optimise_model(params, simulations, model)
