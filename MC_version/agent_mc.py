import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.init as init
from worker import get_batch
import sys
import os
import matplotlib.pyplot as plt

import evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Policy(nn.Module):
    def __init__(self, nemesis_model=None, minimax_evaluation=False):
        """
        nemesis_model : enemy model used for train
        """
        super().__init__()

        self.device = device

        self.nemesis_model = nemesis_model

        # value
        self.value_input_dim = 32
        self.value_hidden_dim = 40
        self.value_output_dim = 1
        self.value_number_hidden_layer = 10  # previously 40

        self.fc_layers = nn.ModuleList()

        prev_dim = self.value_input_dim

        for layer in range(self.value_number_hidden_layer):
            self.fc_layers.append(nn.Linear(prev_dim, self.value_hidden_dim))
            self.fc_layers.append(nn.ReLU())  # previously ReLu (Sigmoid in article)
            prev_dim = self.value_hidden_dim

        self.fc_layers.append(nn.Linear(prev_dim, self.value_output_dim))
        # self.fc_layers.append(nn.Sigmoid())  # previously without (in article)

        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)  # Xavier/Glorot initialization
                init.constant_(layer.bias, 0.0)  # Initialize bias to zero

        self.to(self.device)

        self.first_turn_of_model = 0  # [!] use it many times in code need to be global or other
        self.color_of_model = self.first_turn_of_model
        self.minimax_evaluation = minimax_evaluation  # if True : evaluation use minimax (impact on training when metrics collected)
        self.depth_minimax = 1

    def evaluate_value(self, state):
        x = state.float()
        for layer in self.fc_layers:
            x = layer(x)
        return x

    def minimax_value(self, env, depth=3):
        """
        return value of state of env according to minimax algo
        depth>=0
        """
        # no draw because no truncate
        if depth == 0:  # leaf
            # print("depth == 0")
            return self.evaluate_value(torch.from_numpy(env.get_state()).to(device))
        if env.is_over():
            # print("en_is_over")
            color_looser_player = env.active
            if color_looser_player == self.color_of_model:
                return 0
            else:
                return 1
        if env.active == self.color_of_model:  # maximizing
            # print("max depth", depth)
            value = -1_000_000  # -infinity
            moves, states = env.available_states()
            for move in moves:
                value = max(value, self.minimax_value(env.peek_move(move), depth - 1))
        else:  # minimizing
            # print("min depth", depth)
            value = 1_000_000  # infinity
            moves, states = env.available_states()
            for move in moves:
                value = min(value, self.minimax_value(env.peek_move(move), depth - 1))

        return value

    def get_index_to_act_from_minimax(self, env):
        """
        supppse env is not end
        :return: index of the chosen action
        """
        # [!] can not work because function return index not corresponding to the sent available_states in get_index_to_act
        if env.is_over():
            raise RuntimeError("game not suppose to be over")

        moves, states = env.available_states()

        idx = 0
        index_to_act = 0
        max_value = 0
        for move in moves:
            peek_env = env.peek_move(move)
            value = self.minimax_value(peek_env, depth=self.depth_minimax - 1)
            if value > max_value:
                max_value = value
                index_to_act = idx

            idx += 1

        return index_to_act

    def get_move_to_act(self, env):
        """
        :param available_states: python list of available state
        :return: move
        """

        if not self.minimax_evaluation:
            moves, states = env.available_states()
            available_states = torch.from_numpy(np.array(states)).to(self.device)

            values = self.evaluate_value(available_states)
            index = torch.argmax(values)

            return moves[index]

        else:  # MINIMAX
            return self.get_index_to_act_from_minimax(env)

    def train(self, num_epochs=100, learning_rate=0.01, number_of_parties_for_batch=100, plot=False):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        # metrics
        losses = []
        win_means = []
        lose_means = []
        draw_means = []

        for epoch in range(num_epochs):
            # generate data

            states, rewards = get_batch(number_of_parties_for_batch, self, nemesis_model=self.nemesis_model)

            # mean_reward = float(rewards.mean())
            # if max_mean_rewards < mean_reward:
            #     max_mean_rewards = mean_reward
            #     torch.save(self.state_dict(), "model_save/model_" + str(mean_reward) + ".pt")
            #
            # mean_rewards.append(mean_reward)

            values = self.evaluate_value(states)
            loss = criterion(values.squeeze(), rewards)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            losses.append(loss)

            if (epoch + 1) % 10 == 0:
                self.save()
                parties_lose_mean, parties_win_mean, parties_draw_mean = evaluate.evaluate_against_random(self,
                                                                                                          number_of_parties=20)
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}, Win: {parties_win_mean:.4f}, Draw : {parties_draw_mean}, Lose : {parties_lose_mean}')
                win_means += [parties_win_mean]
                lose_means += [parties_lose_mean]
                draw_means += [parties_draw_mean]

        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(losses, label='Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Epochs')
            plt.legend()
            plt.grid(True)
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.plot(losses, win_means, label='wins')
            plt.plot(losses, lose_means, label='loses')
            plt.plot(losses, draw_means, label='draws')

            plt.xlabel('Epochs')
            plt.ylabel('metrics')
            plt.title('win, lose and draw Over Epochs')
            plt.legend()
            plt.grid(True)
            plt.show()

    def save(self, name='model_test_init'):
        torch.save(self.state_dict(), 'model_save/' + name + '.pt')

    def load(self, name='model_80p'):
        self.load_state_dict(torch.load('model_save/' + name + '.pt', map_location=self.device))

    def load_absolute(self, name):
        self.load_state_dict(torch.load(name, map_location=self.device))


if __name__ == '__main__':
    # nemesis_model = Policy()
    # nemesis_model.load()
    #
    # policy = Policy(nemesis_model=nemesis_model)

    policy = Policy()
    # policy.load()
    # print("model load")
    policy.train(num_epochs=5000, number_of_parties_for_batch=1,
                 plot=True)  # previously number_of_parties_for_batch=100
