from checker_env_MC import CheckerEnv
from numpy import random
import torch
import numpy as np


def get_batch(number_of_parties, model, gamma=0.99, verbose=False, show_env=False, nemesis_model=None):
    """
    return batch of value of states with model against random policy
    :param number_of_parties: number of parties
    :param model: model to play with
    :param gamma: discount factor
    :param verbose: verbose
    :param show_env: show environment
    :param nemesis_model: nemesis model to play against (if None : play against random policy)
    :return:
    """

    # for party in range(number_of_parties):
    states_visited = []  # states visited by the model
    rewards = []
    env = CheckerEnv()

    first_turn_of_model = 0  # [!] use it many times in code need to be global or other
    color_of_model = first_turn_of_model
    parties_win = 0

    for party in range(number_of_parties):

        state, cur_player = env.reset()
        done = False
        total_reward = 0
        turn = 0

        intermediate_rewards = []
        cumulative_gamma = 1

        number_new_states = 0

        if show_env:
            print(env)

        while not done:
            moves, states = env.available_states()

            # choose action
            if turn % 2 == first_turn_of_model:
                action = model.get_move_to_act(env)
            else:
                if nemesis_model:
                    action = nemesis_model.get_move_to_act(env)

                else:
                    action = random.choice(moves)
            state, reward, terminated, truncated, cur_player = env.step(action)
            done = terminated or truncated

            # save state if it was the result of the model action
            if turn % 2 == first_turn_of_model:
                states_visited.append(state)
                number_new_states += 1
                intermediate_rewards.append(cumulative_gamma)
                cumulative_gamma *= gamma

            if not env.jump:  # if not in jump session add a turn
                turn += 1
            if show_env:
                print(env)

        if verbose:
            print(" --- total turn ---", turn)

        color_looser_player = env.active

        if truncated:  # draw : reward = 0.5
            intermediate_rewards.reverse()
            intermediate_rewards = [value * 0.5 for value in intermediate_rewards]

            rewards += intermediate_rewards
        elif color_looser_player == color_of_model:  # lose : reward = 1
            rewards += [0] * number_new_states
        else:  # win : reward = 0
            intermediate_rewards.reverse()
            rewards += intermediate_rewards
            parties_win += 1

    states_torch, rewards_torch = torch.from_numpy(np.array(states_visited)).to(model.device), torch.tensor(rewards).to(
        model.device).float()

    if verbose:
        print("get_batch -> win rate: ", parties_win / number_of_parties)

    return states_torch, rewards_torch


if __name__ == '__main__':
    from agent_mc import Policy

    policy = Policy()
    policy.load("model_80p", )
    number_of_parties = 20
    states, rewards = get_batch(number_of_parties, policy, nemesis_model=None, verbose=True)
