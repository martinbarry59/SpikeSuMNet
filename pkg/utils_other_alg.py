# coding: utf-8
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
# from neuronpy.graphics import spikeplot
import scipy
import tqdm
import os
import random
from sklearn.feature_extraction import image
import warnings
warnings.filterwarnings('ignore')
#from utils import *
def p_hat(x, number_rooms):
    return (1 + (np.arange(number_rooms) == x))
def gammaln(a):
    return np.log(sc_gamma(a))
def DIR(alpha,beta):
    return (gammaln(np.sum(alpha)) - gammaln(np.sum(beta)) - np.sum(gammaln(alpha)) +
            np.sum(gammaln(beta)) + np.sum((alpha - beta) * (digamma(alpha) - digamma(np.sum(alpha)))))
def HMM(epochs = 10000, alpha_states = 2,alpha_output = 16, change_prob = 0.0005, eta = 0.005, simulation = None):
    # print('H: ',change_prob,'  eta: ',eta)
    A = np.ones((alpha_states, alpha_states)) / alpha_states
    A = [[1-change_prob, change_prob], [change_prob, 1-change_prob]];
    #A = [[0.995, 0.005], [0.005, 0.995]];
    B = np.ones((alpha_output, alpha_output, alpha_states)) / alpha_output
    q = [0,1]#np.ones(alpha_states) / alpha_states
    phi = np.zeros((alpha_states, alpha_states, alpha_states, alpha_output,alpha_output))
    maze = 0
    epsilon = 1e-4
    temp = np.zeros_like(phi)
    temp_q = np.zeros_like(q)
    dic_evolution  = {}
    dic_evolution['error'] = []
    log_likelihood_error = 0
    old_output = 0
    new_output = 1

    for room in range(epochs -1):
        old_output, new_output, maze = simulation['rooms'][room],simulation['rooms'][room+1], simulation['maze'][room+1]
        Z = np.sum(q * np.matmul(A,B[new_output, old_output,:]))
        gamma = A * B[new_output, old_output, :] / Z


        phi_old = phi.copy()
        G = init_G(q, alpha_states)
        Delta = np.zeros((alpha_states,alpha_states,alpha_states,alpha_states,alpha_output,alpha_output))
        for k in range(alpha_output):
            for k_old in range(alpha_output):
                    Delta[:,:,:,:,k,k_old] =  G * (old_output == k_old) * (new_output == k)

        temp_no_loop = np.zeros_like(phi_old)
        for h in range(alpha_states):
            new = np.zeros_like(temp_no_loop[:,:,h,:,:])
            for l in range(alpha_states):
                new += gamma[l,h]* (phi_old[:,:,l,:,:]+ eta * (Delta[:,:,h,l,:,:] - phi_old[:,:,l,:,:]))
            temp_no_loop[:,:,h,:,:] = new
        phi = temp_no_loop.copy()

        q = np.matmul(gamma.T, q)

        if room > 3000:
            A = np.sum(np.sum(np.sum(phi, axis = 4),axis =3), axis = 2 ) / np.sum(np.sum(np.sum(np.sum(phi, axis = 4),axis =3), axis = 2 ), axis = 1)
            B = np.sum(np.sum(phi, axis = 2), axis = 0 ).T / np.sum(np.sum(np.sum(np.sum(phi, axis = 4),axis =3), axis = 2 ), axis = 0) + epsilon
        transition = B.copy()
        row_sums = transition.sum(axis=0)
        B = transition / row_sums[np.newaxis,:]
        all_possible = np.matmul(B,q)
        transition = all_possible
        row_sums = transition.sum(axis=1)
        T = transition / row_sums[:, np.newaxis]
        current_T_matrix = simulation['transitions'][maze]
        dic_evolution['error'] += [np.mean((current_T_matrix-T)**2)]
    return dic_evolution, np.mean(dic_evolution['error'][3000:])
def init_G(q,alpha_states):
    G = np.zeros((alpha_states,alpha_states, alpha_states, alpha_states))
    for i in range(alpha_states):
        for j in range(alpha_states):
            for h in range(alpha_states):
                for l in range(alpha_states):
                    G[i,j,h,l] = (i==l) * (j==h) * q[l]
    return G
def Smile(epochs, m, number_rooms, simulation):
    pi = np.ones((number_rooms,number_rooms))
    # print('m: ',m)
    dic_evolution = {}
    dic_evolution['error'] = []
    dic_evolution['gamma'] = []
    dic_evolution['Transitions'] = []
    epsilon = 1e-10
    log_likelihood_error = 0
    for room in range(epochs):
        old_output, new_output , maze = simulation['rooms'][room],simulation['rooms'][room + 1],simulation['maze'][room + 1]
        S =DIR(pi[:, old_output], p_hat(new_output, number_rooms))
        bmax = DIR(p_hat(new_output, number_rooms), pi[:, old_output])
        gamma = m * S/(1 + m * S)
        pi[:, old_output] = pi[:, old_output] * (1 - gamma) + (p_hat(new_output, number_rooms) * gamma)
        row_sums = (pi-1 + epsilon).sum(axis=1)
        T = (pi - 1 + epsilon)/ row_sums[:, np.newaxis]
        current_T_matrix = simulation['transitions'][maze]
        dic_evolution['error'] += [np.mean((current_T_matrix-T)**2)]
        dic_evolution['gamma']  += [gamma]
    dic_evolution['Transitions'] = current_T_matrix
    return dic_evolution, np.mean(dic_evolution['error'])
def VarSmile(epochs, m,epsilon, number_rooms, simulation):
    Proba = (np.zeros((number_rooms,number_rooms)) + epsilon) / (np.zeros((number_rooms,number_rooms)) + number_rooms * epsilon)
    Count = np.zeros((number_rooms,number_rooms))
    # print('m: ',m)
    dic_evolution = {}
    dic_evolution['error'] = []
    dic_evolution['gamma'] = []
    dic_evolution['Transitions'] = []
    log_likelihood_error = 0
    for room in range(epochs-1):
        old_output, new_output , maze = simulation['rooms'][room],simulation['rooms'][room + 1],simulation['maze'][room + 1]
        Sbf =p_hat(new_output, number_rooms)/ Proba[new_output, old_output] ## P_hat flat prior
        gamma = m * Sbf/(1 + m * Sbf)
        ds = np.zeros(number_rooms)
        ds[new_output] = 1
        Count[:, old_output] = Count[:, old_output] * (1 - gamma) + ds
        Proba[:, old_output] = (Count[:, old_output] + epsilon) / (np.sum(Count[:, old_output]) + number_rooms * epsilon)
        row_sums = Proba.sum(axis=1)
        T = Proba/ row_sums[:, np.newaxis]
        current_T_matrix = simulation['transitions'][maze]
        dic_evolution['error'] += [np.mean((current_T_matrix.cpu().numpy()-T)**2)]
        dic_evolution['gamma']  += [gamma]
    dic_evolution['Transitions'] = Count
    return dic_evolution, np.mean(dic_evolution['error'])
