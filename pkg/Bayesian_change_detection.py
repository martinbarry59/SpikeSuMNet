import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tqdm
from scipy.special import gamma

## Dirichlet probability for a room st x = [0,0,...,1,0,..,0]
def lgamma(x):
    return np.log(gamma(x))


def P(x, alpha):
    
    alpha0 = np.sum(alpha[:],axis = 0)
    tmp = lgamma(alpha[x] + 1) - lgamma(alpha[x]) + lgamma(alpha0) - lgamma(alpha0 + 1)
    return np.exp(tmp)


def approximate(f_probabilities,alpha,cutoff):
    alpha = alpha[:,:,f_probabilities > cutoff]
    f_probabilities = f_probabilities[f_probabilities > cutoff]
    
    f_probabilities /= np.sum(f_probabilities)
    return f_probabilities, alpha


def mean(x):
    return  x / np.sum(x)


def Bayesian_change_point(epochs = 100, n_room = 16,rmax = 1000, s = 1e-0, pc = 0.001, simulation = None, plot = False):

    # print('s: ',s,'  lambda: ',lambda_)
    

    dic_evolution = {}
    dic_evolution['error'] = []
    dic_evolution['T'] = []
    prediction = np.zeros(n_room)
#     list_probabilities = n_room * [[1.]]
#     list_posterior = n_room * [n_room * [[ s]]]
    f_probabilities = np.array([1.])
    f_posterior_all = s * np.ones((n_room, n_room, 1))
    T = np.zeros((n_room, n_room))
    run_time = []
    log_likelihood_error = 0

    for epoch in  tqdm.tqdm(range(epochs-1),disable = (1 - plot)):
        x_old, x, maze = simulation['rooms'][epoch],simulation['rooms'][epoch + 1],simulation['maze'][epoch + 1]
        f_posterior = f_posterior_all[:,x_old,:]
#         f_posterior = np.array(list_posterior[x_old])
#         f_probabilities = np.array(list_probabilities[x_old])

#         pi_t = P(x, f_posterior)       # P(x_t|νʳ, χʳ)
        pi_t = P(x, f_posterior) 
        
        f_probabilities *= pi_t
        f_probabilities /= sum(f_probabilities)
        
        f_posterior[x, :] += 1
        f_posterior_all[:,x_old,:] = f_posterior.copy()
        f_probabilities, f_posterior_all = approximate(f_probabilities, f_posterior_all, 1e-16)
#         parameter_estimate = np.sum(mean(f_posterior[:,:]) * f_probabilities, axis = 1)
        parameter_estimate = np.sum(mean(f_posterior_all[:,x_old,:]) * f_probabilities, axis = 1)
        
        p_switch = np.sum(f_probabilities) * pc     # ∝ P(r_t = 0, x_{1:t})
        
        # @show sum(f.probabilities)
        f_probabilities *= (1 - pc)              # ∝ P(r_t = 1, x_{1:t})
        f_probabilities = np.append(f_probabilities, p_switch)
#         tmp = np.ones((n_room, np.shape(f_posterior)[-1] + 1))
#         tmp[:,:-1] = f_posterior
#         tmp[:,-1] = s
        tmp = np.ones((n_room, n_room, np.shape(f_posterior_all)[-1] + 1))
        tmp[:,:,:-1] = f_posterior_all.copy()
        tmp[:,:,-1] = s
        f_posterior_all = tmp.copy()
        
        T[x_old, :] = parameter_estimate / np.sum(parameter_estimate)
        current_T_matrix = simulation['transitions'][maze]
        dic_evolution['error'] += [np.mean((current_T_matrix.cpu().numpy()-T)**2)]
#         list_probabilities[x_old] = f_probabilities.copy()
#         list_posterior[x_old] = f_posterior.copy()
    dic_evolution['T'] = T.copy()

    return dic_evolution, np.mean(dic_evolution['error'] ), np.array(run_time).T
