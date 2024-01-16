import torch
import torch.nn as nn
import torch.optim as optim

#Define Neural Network
class ValueNetwork(nn.Module):


    def __init__(self):

        super(ValueNetwork, self).__init__()
        # 2 different models to test (uncomment ONLY one of the 2 following) :

        # complex model

        ##
        # self.fc1 = nn.Linear(32, 200)
        # self.activation = nn.Tanh()
        # self.fc2 = nn.Linear(200, 200)
        # self.fc3 = nn.Linear(200, 300)
        # self.fc4 = nn.Linear(300, 200)
        # self.fc5 = nn.Linear(200, 100)
        # self.fc6 = nn.Linear(100, 1)
        ##

        # simple model

        ##
        self.fc1 = nn.Linear(32, 100)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(100, 1)
        ##

        ###end##

        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)


    def predict(self, x):

        # simple model

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        #complex model (uncomment to have the complex model)

        # x = self.activation(x)
        # x = self.fc3(x)
        # x = self.activation(x)
        # x = self.fc4(x)
        # x = self.activation(x)
        # x = self.fc5(x)
        # x = self.activation(x)
        # x = self.fc6(x)
        # x = self.sigmoid(x)
        return x


    def update(self, output, target_tensor):


        loss = self.criterion(output, target_tensor)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def create_ANN(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        value_net = self.to(device)
        return value_net
