import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.init as init
from worker import get_batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import matplotlib.pyplot as plt

class Policy(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = device

        # value
        self.value_input_dim = 32
        self.value_hidden_dim = 32
        self.value_output_dim = 1
        self.value_number_hidden_layer = 40

        self.fc_layers = nn.ModuleList()

        prev_dim = self.value_input_dim

        for layer in range(self.value_number_hidden_layer):
            self.fc_layers.append(nn.Linear(prev_dim, self.value_hidden_dim))
            self.fc_layers.append(nn.ReLU())
            prev_dim = self.value_hidden_dim

        self.fc_layers.append(nn.Linear(prev_dim, self.value_output_dim))

        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)  # Xavier/Glorot initialization
                init.constant_(layer.bias, 0.0)  # Initialize bias to zero

        self.to(self.device)

    def evaluate_value(self, state):
        x = state.float()
        for layer in self.fc_layers:
            x = layer(x)
        return x

    def get_index_to_act(self, available_states):
        """
        :param available_states: python list of available state
        :return: index of the chosen action
        """
        available_states = torch.from_numpy(np.array(available_states)).to(self.device)
        values = self.evaluate_value(available_states)

        return torch.argmax(values)

    def train(self, num_epochs=100, learning_rate=0.01, plot=False):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        number_of_parties_for_batch = 100

        losses = []

        for epoch in range(num_epochs):
            # generate data

            states, rewards = get_batch(number_of_parties_for_batch, self)

            values = self.evaluate_value(states)
            loss = criterion(values.squeeze(), rewards)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            losses.append(loss)

            if (epoch + 1) % 3 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}')

        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(losses, label='Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Epochs')
            plt.legend()
            plt.grid(True)
            plt.show()

    # def save(self):
    #     torch.save(self.state_dict(), 'model_save/model.pt')

    # def load(self):
    #     self.load_state_dict(torch.load('model_save/model.pt', map_location=self.device))


if __name__ == '__main__':
    policy = Policy()
    policy.train(plot=True)
