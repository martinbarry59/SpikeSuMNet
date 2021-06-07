import os
import torch
import matplotlib.pyplot as plt
import plotly
import math
import random
import plotly.graph_objects as go
def plotly_plot(datas,legend,file = None, plot = False):
    
    fig = go.Figure()
    for data in datas:
        x = torch.arange(len(data))
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
def save(data=None, file=None, type_=None):
    """
    Saving file plotly, matplotlib or npy

    data: the variables to save

    file: relative path of the file, will save it in a results folder

    type_: give the type of the file to save (fig_ly, fig,data)

    """
    directory = "../results/" + type_ + "/" + file

    if not os.path.exists("/".join(directory.split("/")[:-1])):
        os.makedirs("/".join(directory.split("/")[:-1]))

    if type_ == "fig_ly":
        plotly.offline.plot(data, filename=directory + ".html")
    elif type_ == "fig":
        plt.savefig(directory + ".eps")
    elif type_ == "data":
        torch.save(directory, data)
    else:
        print("NO FILE SPECIFIED: NOT SAVED")


## Environment function functions ##
def create_trans_matrix(maze, movements_list, symmetric=False):
    """
    maze: array of size N

    movements_list: possible moves within a maze

    """
    N = len(maze)
    trans_matrix = torch.zeros((N, N))

    for room in range(N):
        next_rooms_indices = [ (x+maze.index(room))%N for x in movements_list]
        next_rooms = [maze[i] for i in next_rooms_indices]

        for j in next_rooms:
            if symmetric == True:
                trans_matrix[room, j] = 1 / len(next_rooms)
            else:
                trans_matrix[room, j] = torch.random.rand()
    return trans_matrix


def create_move_list(len_maze, number_of_move):
    """
    len_maze:  number of rooms in the maze

    number_of_move: possible moves within a maze

    """
    moves = [1, int(math.sqrt(len_maze))]
    movements_list = [None] * 2 * number_of_move#torch.zeros(2 * number_of_move, dtype=int)
    for i in range(number_of_move):
        ## if only 2 moves chose up and down
        if i < len(moves):
            movements_list[2 * i : 2 * i + 2] = [moves[i], -moves[i]]
        ## if only more than 2 moves chose randomly a new direction
        else:
            possible_move = [x for x in range(1, len_maze) if x not in movements_list]
            new_move = random.choice(possible_move)
            movements_list[2 * i : 2 * i + 2] = [new_move, -new_move]
    
    return movements_list


## Dirichlet functions for maze creation##
def p_hat(x, number_rooms):
    return 1 + (torch.arange(number_rooms) == x)


def gammaln(a):
    return torch.log(sc_gamma(a))


def DIR(alpha, beta):
    return (
        gammaln(torch.sum(alpha))
        - gammaln(torch.sum(beta))
        - torch.sum(gammaln(alpha))
        + torch.sum(gammaln(beta))
        + torch.sum((alpha - beta) * (digamma(alpha) - digamma(torch.sum(alpha))))
    )

def nice_print(params):
    print('---Network parameters---')
    
    for key, values in sorted(params.items(),key = lambda x: (x[0].lower(), x[1])):
        if isinstance(values,dict):
            print(key,':')
            for kkey, vvalues in sorted(values.items(),key = lambda x: (x[0].lower(), x[1])):
                print('   {0}: {1}'.format(kkey,vvalues))
        else:
            print('{0}: {1}'.format(key,values))
        print('------------------------')
def draw_Dirichlet_maze(len_maze, s):
    """
    len_maze:  number of rooms in the maze

    s: Stochasticity of the maze
    """
    T = torch.zeros((len_maze, len_maze))
    T = torch.random.dirichlet(s * torch.ones(len_maze), len_maze)
    return T


def draw_new_room(T, x):
    """
    T: Transition matrix

     old_room:  previous room observed (int)
    """
    CDF = torch.cumsum(T[x],dim=0)
    test = torch.rand(1)

    return torch.argmax(((CDF - test) > 0).float())


## Running functions ##
def make_step(old_room, maze, transitions, volatility, movements_list, Dirichlet=0, maze_presentations = {}):
    """
    move the agent to the new room

    old_room:  previous room observed (int)

    maze: array of size N

    i: current epoch

    epochs: maximum number of epochs used for a full simulation

    mazes: list of possible mazes to switch to

    """
    maze_presentations[maze] = maze_presentations[maze] + 1 if (maze in maze_presentations) else 1
    if Dirichlet != 0:
        new_room = draw_new_room(transitions, old_room)
        if torch.rand(1) < volatility:
            maze += 1

    else:

        #         N = len(mazes[maze])
        if torch.rand(1) < volatility and maze_presentations[maze] > int(1/volatility):
            current_maze = maze
            while maze == current_maze:
                maze = random.randint(0,len(transitions)-1)

        new_room = draw_new_room(transitions[maze], old_room)
  
    return new_room, maze, maze_presentations


def create_simulation(
    epochs=100000,
    number_rooms=16,
    volatility=0.001,
    n_moves=2,
    n_maze=4,
    seed=None,
    Dirichlet=0,
    deter_start=None,
    symmetric=False,
):
    """
    epochs: Number of steps to run simulation

    number_rooms: length of the maze (must be a perfect square)

    volatility: Probability to switch maze

    n_moves: number of possible movements from one room

    n_maze: number of initialised maze

    seed: reproducibility parameter

    Dirichlet: Allows to draw more complex dirichlet maze, if not 0 Dirichlet corresponds to the stochasticity of the maze

    deter_start: dictionnary (t_end, n_maze) allows a deterministic start for a number of epoch(t_end) allowing transition to only some mazes(n_mazes)
    """
    if seed:
        torch.manual_seed(seed)
    ## Do the simulation ##

    ## create N mazes ##

    simulation = {}
    movements_list = create_move_list(number_rooms, n_moves)
    if Dirichlet == 0:

        maze_1 = list(torch.arange(number_rooms))
        mazes = [maze_1]
        transitions = [create_trans_matrix(maze_1, movements_list, symmetric=symmetric)]

        for _ in range(n_maze - 1):
            maze = list(torch.arange(number_rooms))
            random.shuffle(maze)
            mazes += [maze]
            transitions += [
                create_trans_matrix(maze, movements_list, symmetric=symmetric)
            ]
    else:
        mazes = draw_Dirichlet_maze(number_rooms, Dirichlet)
        transitions = [mazes.copy()]

    ## save all the datas within one big dictionnary ##
    simulation["mazes"] = mazes
    simulation["transitions"] = transitions
    simulation["epochs"] = epochs
    simulation["number_rooms"] = number_rooms
    simulation["Dirichlet"] = Dirichlet
    simulation["volatility"] = volatility
    simulation["n_moves"] = n_moves
    simulation["change_points"] = {}
    simulation["change_points"][0] = 0
    new_room = 0
    maze = 0
    simulation["rooms"] = [new_room]
    simulation["maze"] = [maze]
    current_maze = 0

    deter_switch_max_step = epochs if deter_start is None else deter_start["tend"]
    maze_presentations = {}
    for epoch in range(epochs):
        N_mazes = (
            len(mazes)
            if deter_start is None
            else (
                (epoch < deter_switch_max_step) * deter_start["n_maze"]
                + (epoch >= deter_switch_max_step) * len(mazes)
            )
        )
        if volatility == 0 or (
            deter_start is not None and (epoch <= deter_start["tend"])
        ):

            for j in range(N_mazes):

                if (
                    epoch >= j * deter_switch_max_step / N_mazes
                    and epoch < ((j + 1) * deter_switch_max_step) / N_mazes
                ):
                    maze = j
            new_room, maze,maze_presentations = make_step(
                new_room,
                maze,
                simulation["transitions"],
                0,
                movements_list,
                Dirichlet=Dirichlet,
                maze_presentations = maze_presentations,
            )
        else:
            new_room, maze,maze_presentations = make_step(
                new_room,
                maze,
                simulation["transitions"],
                volatility,
                movements_list,
                Dirichlet=Dirichlet,
                maze_presentations = maze_presentations,
            )
        if current_maze != maze:
            simulation["change_points"][epoch] = maze
            current_maze = maze
            if Dirichlet != 0:
                mazes = draw_Dirichlet_maze(len(mazes[0]), Dirichlet)
                simulation["mazes"] = mazes
                simulation["transitions"] += [mazes.copy()]
                mazes = draw_Dirichlet_maze(len(mazes[0]), Dirichlet)

        simulation["rooms"] += [new_room]
        simulation["maze"] += [maze]

    return simulation


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


def plot_spike_train(spikes_mat,neurons,n_memory,title = None,directory = None):
    
    linewidth = 0.8
    
    event_times, event_ids = torch.where(spikes_mat.T)
    event_times = event_times[:-1]
    event_ids = event_ids[:-1]

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005


    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(figsize=(8, 8))
    if title is not None:
        plt.title(title)
    n_t, n_n = spikes_mat.T.shape
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in')
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    if n_memory > 1:
        ax_histy = plt.axes(rect_histy)
        ax_histy.tick_params(direction='in', labelright=True, labelleft=True)

    # the scatter plot:
    for n, t in zip(event_ids,event_times):
        ax_scatter.vlines(t, n - 0.25, n + 0.25, linewidth=linewidth)

    # now determine nice limits by hand:
    binwidth = 0.25
    ax_scatter.set_ylim([0 , n_n])
    ax_scatter.set_xlim([0, n_t ])
    spikes = spikes_mat.reshape(n_memory,neurons,-1)
    sample = torch.sum(spikes,dim =(0,1))
    ax_histx.bar(torch.arange(spikes.shape[-1]), sample / torch.sum(sample))
    sample = torch.sum(spikes,dim =(1,2))
    if n_memory > 1:
        ax_histy.barh(torch.arange(n_memory), sample / torch.sum(sample))
        ax_histy.yaxis.set_label_position("right")
        ax_histy.yaxis.tick_right()
    plt.ylim([-0.45,n_memory -1 + .45])
    plt.show()
