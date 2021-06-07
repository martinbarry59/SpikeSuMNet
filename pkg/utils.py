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
from scipy.special import gamma as sc_gamma
from scipy.special import digamma
import plotly.graph_objects as go
import plotly
warnings.filterwarnings('ignore')
def find_parameters(s):
    arr = s.split('_')
    if '.pkl' in s or '.npy' in s:
        return float(arr[3]), float(arr[5]), float(arr[8][:-4])
    else:
        return float(arr[3]), float(arr[5]), float(arr[8])
def plotly_plot(data,legend,file = None, plot = False):
    x = np.arange(len(data))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=data,mode='lines'))
    layout = go.Layout()
    fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    legend=go.layout.Legend(

        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=18,
            color="black"
        ),
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text=legend[0],
            font=dict(
            family="sans-serif",
            size=18,

            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text=legend[1],
            font=dict(
                family="sans-serif",
                size=18,
            )
        )
    )
)
    if file:
        save(data = fig, file = file, type_= 'fig_ly')
    if plot:
        fig.show()
def save(data = None, file = None, type_ = None):
    directory = 'results/'+type_+'/'+file
    if not os.path.exists('/'.join(directory.split('/')[:-1])):
        os.makedirs('/'.join(directory.split('/')[:-1]))
    if type_ == 'fig_ly':
        plotly.offline.plot(data, filename=directory+'.html', auto_open=False)
#         data.write_image("images/fig1.pdf")
    elif type_ == 'fig':
        plt.savefig(directory+'.eps')
    elif type_ == 'data':
        np.save(directory,data)
    else :
        print('NO FILE SPECIFIED: NOT SAVED')
def create_move_list(len_maze, number_of_move, symm = True):
    moves = [1, int(np.sqrt(len_maze))]
    if symm:
        movements_list = np.zeros(2 * number_of_move, dtype = int)
    else:
        movements_list = np.zeros(number_of_move, dtype = int)
    for i in range(number_of_move):
        if i < len(moves):
            if symm:
                movements_list[2 * i:2 * i + 2] += [moves[i], -moves[i]]
            else:
                movements_list[i:i + 1] += [moves[i]]
        else:
            x = [x for x in range(1, len_maze) if x not in movements_list]
            r = random.choice(x)
            if symm:
                movements_list[2 * i:2 * i + 2] += [r, -r]
            else:
                movements_list[i:i + 1] += [r]
    return movements_list
def draw_Dirichlet_maze(n_rooms, s):
    T = np.zeros((n_rooms,n_rooms))
    T = np.random.dirichlet(s * np.ones(n_rooms), n_rooms)
    return T
def draw_new_room(T, x):
    idx = np.arange(len(T[x]))
    np.random.shuffle(idx)
    proba = 0
    i = 0
    while True:
        if np.random.rand() < proba + T[x][idx[i]]:
            return idx[i]
        else:
            proba += T[x][idx[i]]
            i += 1
def make_step(new_room, maze, i, epochs, mazes,change_prob, movements_list,file = None, Dirichlet = 0):

    old_room = new_room

    if Dirichlet != 0:
        new_room = draw_new_room(mazes, new_room)
        if np.random.rand()< change_prob:
            maze += 1
            
    else:
        number_rooms = len(mazes[maze])
        number_rooms = len(mazes[maze])
        if np.random.rand()< change_prob:
            old_maze = maze
            while (maze == old_maze):
                maze = np.random.randint(len(mazes))
            if file:
                file.write('{1}, currently in maze {0}'.format(maze, i)+'\n')

        jump  = movements_list[np.random.randint(0,len(movements_list))]
        if change_prob > 0:
            new_room = mazes[maze][(mazes[maze].index(old_room)+jump)%number_rooms]

        else:
            for j in range(len(mazes) ):
                if (i>= j * epochs / len(mazes) and i< ((j +1)* epochs ) / len(mazes)):
                    new_room =mazes[j][(mazes[j].index(old_room)+jump)%number_rooms]
                    maze = j
    return old_room, new_room, maze
def p_hat(x, number_rooms):
    return (1 + (np.arange(number_rooms) == x))
def gammaln(a):
    return np.log(sc_gamma(a))
def DIR(alpha,beta):
    return (gammaln(np.sum(alpha)) - gammaln(np.sum(beta)) - np.sum(gammaln(alpha)) +
            np.sum(gammaln(beta)) + np.sum((alpha - beta) * (digamma(alpha) - digamma(np.sum(alpha)))))
def create_trans_matrix(maze, list_):
    trans_matrix = np.zeros((len(maze), len(maze)))

    for steps in range(len(maze)):
        b = (maze.index(steps) + list_) % len(maze)
        reach = [ maze[i] for i in b]

        for j in reach:
            trans_matrix[steps, j] = 1. / len(b)
    return trans_matrix
def create_simulation(epochs = 100000, number_rooms =16, change_prob = 0.001, n_moves = 2, n_maze = 4, seed = None, Dirichlet = 0):

    if seed:
        np.random.seed(seed)
    ## Do the simulation ##

    ## create N mazes ##

    simulation = {}
    movements_list = create_move_list(number_rooms, n_moves)
    if Dirichlet == 0:

        maze_1 = list(np.arange(number_rooms))
        mazes = [maze_1]
        transitions = [create_trans_matrix(maze_1, movements_list)]

        for i in range(n_maze-1):
            maze = list(np.arange(number_rooms))
            np.random.shuffle(maze)
            mazes += [maze]
            transitions += [create_trans_matrix(maze, movements_list)]
    else:
        mazes = draw_Dirichlet_maze(number_rooms, Dirichlet)
        transitions = [mazes.copy()]
    simulation['mazes'] = mazes
    simulation['transitions'] = transitions
    simulation['epochs'] = epochs
    simulation['number_rooms'] = number_rooms
    simulation['Dirichlet'] = Dirichlet
    simulation['change_prob'] = change_prob
    simulation['n_moves'] = n_moves
    new_room = 0
    maze = 0
    simulation['rooms'] = [new_room]
    simulation['maze'] = [maze]
    old_maze = 0
    for epoch in range(epochs):
        old_room, new_room, maze = make_step(new_room, maze, epoch, epochs,simulation['mazes'],change_prob, movements_list,Dirichlet = Dirichlet)
        if Dirichlet != 0 and maze != old_maze:
            mazes = draw_Dirichlet_maze(len(mazes[0]), Dirichlet)
            simulation['mazes'] = mazes
            simulation['transitions'] += [mazes.copy()] 
            mazes = draw_Dirichlet_maze(len(mazes[0]), Dirichlet)
            old_maze = maze
        simulation['rooms'] += [new_room]
        simulation['maze'] +=[maze]
    return simulation
