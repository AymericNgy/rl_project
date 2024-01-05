import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Policy(nn.Module):
    def __init__(self):
        super().__init__()

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

    def evaluate_value(self, state):
        x = state
        for layer in self.fc_layers:
            x = layer(x)
        return x

    def get_index_to_act(self, available_states):
        """
        :param available_states: python list of available state
        :return: index of the chosen action
        """
        values = [self.evaluate_value(state) for state in available_states]

        index_of_max = values.index(max(values))

        return index_of_max

    def train(self):
        pass
    # def save(self):
    #     torch.save(self.state_dict(), 'model_save/model.pt')

    # def load(self):
    #     self.load_state_dict(torch.load('model_save/model.pt', map_location=self.device))



