import os
import torch
import matplotlib.pyplot as plt
import plotly
import math
import plotly.graph_objects as go
import random
import numpy as np
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
                trans_matrix[room, j] = torch.rand(1)
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
            possible_move = [x for x in range(1, int(len_maze / 2)) if x not in movements_list]
            new_move = random.choice(possible_move)
            movements_list[2 * i : 2 * i + 2] = [new_move, -new_move]
    if number_of_move == 0:
        movements_list = [1]
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
                maze = torch.randint(len(transitions),(1,)).item()

        new_room = draw_new_room(transitions[maze], old_room)
  
    return new_room, maze, maze_presentations


def create_simulation(
    epochs=1,
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
            maze = torch.arange(number_rooms)
            indexes = torch.randperm(maze.shape[0])
            maze = list(maze[indexes])
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
    new_room = torch.tensor(0)
    
    maze = 0
    simulation["rooms"] = []
    simulation["maze"] = [maze]
    current_maze = 0

    deter_switch_max_step = epochs if deter_start is None else deter_start["tend"]
    maze_presentations = {}
    torch.manual_seed(random.randint(0,1e8))
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
