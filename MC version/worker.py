from checker_env import CheckerEnv
from numpy import random
import torch
import numpy as np
from tqdm import tqdm


def get_batch(number_of_parties, model, verbose=False):
    """
    return batch of value of states with model against random policy
    :param number_of_parties:
    :param model:
    :return:
    """
    # for party in range(number_of_parties):
    states_visited = []  # states visited by the model
    rewards = []
    env = CheckerEnv()

    first_turn_of_model = 0
    color_of_model = first_turn_of_model

    for party in range(number_of_parties):

        state, cur_player = env.reset()
        done = False
        total_reward = 0
        turn = 0

        number_new_states = 0

        while not done:
            moves, states = env.available_states()

            # choose action
            if turn % 2 == first_turn_of_model:
                index = model.get_index_to_act(states)
                action = moves[index]
            else:
                action = random.choice(moves)
            state, reward, done, cur_player = env.step(action)
            # total_reward += reward

            # save state if it was the result of the model action
            if turn % 2 == first_turn_of_model:
                states_visited.append(state)
                number_new_states += 1

            if not env.jump:  # if not in jump session add a turn
                turn += 1

        color_looser_player = env.active
        if color_looser_player == color_of_model:
            final_reward = 0
        else:
            final_reward = 1
        rewards += [final_reward] * number_new_states

    states_torch, rewards_torch = torch.from_numpy(np.array(states_visited)).to(model.device), torch.tensor(rewards).to(
        model.device).float()

    if verbose:
        print("get_batch -> mean reawrd : ", rewards_torch.mean())

    return states_torch, rewards_torch


if __name__ == '__main__':
    from agent_mc import Policy

    policy = Policy()
    number_of_parties = 100
    states, rewards = get_batch(number_of_parties, policy)
    print(rewards.mean())
