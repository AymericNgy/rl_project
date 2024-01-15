from checker_env import CheckerEnv as CheckerEnv1
from plateau_essai2 import CheckerEnvForMCTS as CheckerEnv2
from numpy import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
import time
from mcts_essai2 import mcts
from copy import deepcopy


def play_party(policy, env, first_turn_of_model, color_of_model, show_env=False, verbose=False, nemesis=None):
    state, cur_player = env.reset()

    done = False

    turn = 0

    if show_env:
        print(env)

    mcts_state = CheckerEnv2()

    while not done:

        moves, states = env.available_states()

        # choose action

        if turn % 2 == first_turn_of_model:

            print("nemesis is back")

            action = policy.get_move_to_act(env)

        else:

            if nemesis == None:

                # action = random.choice(moves)

                print("mcts plays")

                action, root = mcts(deepcopy(mcts_state))

            else:

                action = nemesis.get_move_to_act(env)

        state, reward, terminated, truncated, cur_player = env.step(action)

        mcts_state, reward, done, _ = mcts_state.step(action)

        done = terminated or truncated or done

        # total_reward += reward

        if not env.jump:  # if not in jump session add a turn

            turn += 1

        if show_env:
            print(env)

    if verbose:
        print(" --- total turn ---", turn)

    color_looser_player = env.active

    if truncated:

        return "party draw"

    elif color_looser_player == color_of_model:

        return "party lose"

    else:

        return "party win"
