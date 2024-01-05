from checker_env import CheckerEnv
from numpy import random
import torch


class Worker:
    def get_batch(self, number_of_parties, model):
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
            states_visited.append(state)
            done = False
            total_reward = 0
            turn = 0

            number_new_states = 0

            while not done:
                moves, states = env.available_states()
                # choose action
                if turn % 2 == first_turn_of_model:
                    action = moves[model.get_index_to_act(states)]
                else:
                    action = random.choice(moves)
                state, reward, done, cur_player = env.step(action)
                # total_reward += reward

                if turn % 2 == first_turn_of_model:
                    states_visited.append(state)
                    number_new_states += 1

                if not env.jump:  # if not in jump session add a turn
                    turn += 1

            color_looser_player = env.active
            if color_looser_player == color_of_model:
                final_reward = 1
            else:
                final_reward = 0
            rewards += [final_reward] * number_new_states

        return torch.array(states), torch.array(reward)
