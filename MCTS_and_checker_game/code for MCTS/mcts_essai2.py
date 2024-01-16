from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

from copy import deepcopy
import random
from plateau_essai2 import *
import math
import time
from value_network_checkers import *
import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from pyqtgraph.Qt import QtCore, QtGui

# state is an element of Checker_Env and have variables as jump, player ...

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MCTSNode:
    """MCTS Node."""

    def __init__(self, state, parent=None, action=None):
        """Initialize a node."""

        self.state = state # state of the board, a checker_env element
        self.parent = parent
        self.action = action
        self.children = []
        self.untried_actions,_ = self.state.available_states()
        self.n = 0
        self.w = 0
        self.is_terminal = self.state.check_termination()[0]

    @property
    def fully_expanded(self):
        return len(self.untried_actions) == 0

    def expand(self):
        """Pick an untried action, evaluate it, generate the node for the resulting state (also add it to the children) and return it."""

        action = self.untried_actions.pop()

        next_state = self.state.peek_move(action)

        child_node = MCTSNode(next_state, parent=self, action=action)

        self.children.append(child_node)

        return child_node

    def rollout(self):  # classic rollout, 1 rollout with result =+-1
        """Until termination, move randomly. Return the result (winning player)"""

        state = self.state
        done, result = state.check_termination()
        while not done:
            possible_actions,_ = state.available_states()
            action = random.choice(possible_actions)
            state = state.peek_move(action)
            done, result = state.check_termination()
        return result

    def rollout_version_ANN(self,value_net): # rollout with, for each random choice, a test between the prediction of the value network and a threshold of probability
        """Until termination, move randomly. Return the result (winning player)"""

        state = self.state
        done, result = state.check_termination()
        while not done:
            prediction=value_net.predict(torch.tensor(state.get_state(), dtype=torch.float32).to(device))

            if(self.state.active==BLACK):
                if(prediction>=0.75):

                    return "Black"
            else:
                if(prediction<=0.25):

                    return "White"

            possible_actions,_ = state.available_states()
            action = random.choice(possible_actions)
            state = state.peek_move(action)
            done, result = state.check_termination()
        return result

    def rollout_version_ANN_2(self,value_net): # different rollout : before all the random choices we test if the prediction is better than a threshold : just test one time to gain time from rollout_version_ANN
        """Until termination, move randomly. Return the result (winning player)"""

        state = self.state
        done, result = state.check_termination()
        prediction=value_net.predict(torch.tensor(state.get_state(), dtype=torch.float32).to(device))
        if(self.state.active==BLACK):
            if(prediction>=0.55):

                return "Black"
        else:
            if(prediction<=0.45):

                return "White"
        while not done:

            possible_actions,_ = state.available_states()
            action = random.choice(possible_actions)
            state = state.peek_move(action)
            done, result = state.check_termination()
        return result

    def backpropagate(self, result): # classic backpropagation used with rollout
        """Backprop the result of a rollout up to the root node: For each node in the path update the visits and the number of wins"""

        self.n += 1
        if self.parent:
            if result =="White" and self.parent.state.active==WHITE or result =="Black" and self.parent.state.active==BLACK: # active == BLACK or WHITE
                self.w += 1
            else :
                self.w -= 1
            self.parent.backpropagate(result)


    def backpropagate_stochastic(self, result): # used with mcts training version multiple rollout (try to improve the stability of training the network)

        self.n += 1
        if self.parent:
            if self.parent.state.active==WHITE : # active == BLACK or WHITE
                self.w += 1-result

            elif self.parent.state.active==BLACK :
                self.w += result
            self.parent.backpropagate_stochastic(result)


    def traverse(self):
        """Traverse the nodes until an unexpanded one is found or termination is reached"""

        node = self

        while node.fully_expanded and not node.is_terminal:
            node = node.best_uct_child()

        if node.is_terminal:
            return node

        return node.expand()

    def win_ratio(self):
        """Win Ratio of a node"""

        return self.w/self.n

    def uct(self):
        """UCT value of a node"""

        return self.win_ratio() + math.sqrt(2*math.log(self.parent.n)/self.n)

    def best_child(self):
        """Return the best child (the one with the highest win ratio)"""

        best_win_ratio, child = max((self.children[i].win_ratio(), i) for i in range(len(self.children)))


        return self.children[child]

    def best_uct_child(self):
        """Return the best child according to UCT"""

        best_win_ratio, child = max((self.children[i].uct(), i) for i in range(len(self.children)))

        return self.children[child]


def mcts_training_ANN(state,value_net, iters=500):
    # we don't consider the 0-0 parties because we want to win

    #### if you want to do multiple rollout/training uncomment only the first part and the line : leaf.backpropagate_stochastic(real)

    #### if you want to do one rollout/ training, uncomment only the second part and the line : leaf.backpropagate(simulation_result)

    root = MCTSNode(deepcopy(state))
    totaltime=0
    point=0
    for i in range(iters):
        leaf = root.traverse()
        time1=time.time()
        prediction=value_net.predict(torch.tensor(leaf.state.get_state(), dtype=torch.float32).to(device))
        # predict the final value
        # try to improve the robustness of the real prediction by doing multiple rollout (=len)

        ####
        #First part:
        # real=0
        # len=50
        # for i in range(len):
        #     simulation_result = leaf.rollout()
        #     if(simulation_result=="Black"):
        #         real+=1
        # real=real/len
        ####

        #Second part

        simulation_result = leaf.rollout()
        real=0
        if(simulation_result=="Black"):
            real=1

        #####
        totaltime+=time.time()-time1

        real=torch.tensor([real],dtype=torch.float32).to(device)
        point+= abs(prediction-real)
        value_net.update(prediction,real)

        ### Comment one of the options

        leaf.backpropagate(simulation_result) # use it for one rollout
        # leaf.backpropagate_stochastic(real) # use it for multiple rollout


    torch.save(value_net.state_dict(), 'mon_modele.pth')
    print("Total time for 1 mcts_training call : ",totaltime)
    return root.best_child().action, root, point

def mcts_play_ANN(state,value_net, iters=300):

    # here, important to use a threshold to gain time (different ideas, see rollout_version_ANN and rollout_version_ANN_2

    root = MCTSNode(deepcopy(state))
    totaltime=0
    for i in range(iters):
        leaf = root.traverse()
        time1=time.time()
        simulation_result = leaf.rollout_version_ANN_2(value_net)
        totaltime+=time.time()-time1
        leaf.backpropagate(simulation_result)
    print("Total time for 1 mcts_play_ANN call : ",totaltime)
    return root.best_child().action, root


def mcts(state, iters=300):
    root = MCTSNode(deepcopy(state))
    totaltime=0
    for i in range(iters):

        leaf = root.traverse()
        time1=time.time()
        simulation_result = leaf.rollout()
        totaltime+=time.time()-time1
        leaf.backpropagate(simulation_result)
    print("Total time for 1 mcts call : ",totaltime)
    return root.best_child().action, root


class RealTimePlot(QMainWindow):
    def __init__(self):
        super(RealTimePlot, self).__init__()


        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)


        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)


        self.x_data = []
        self.y_data = []
        self.curve = self.plot_widget.plot(pen='r')
        self.plot_widget.setMouseEnabled(x=True, y=True)

        self.frame = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000)  # update the curve every 1s

    def update_plot(self):
        # add new point to graph
        print("graph")
        print(self.frame)
        self.x_data.append(self.frame)
        self.new_point = self.new_point.to('cpu').detach().numpy()
        self.y_data.append(self.new_point[0])

        # update curve on graph
        self.curve.setData(self.x_data, self.y_data)

        #increment absciss axis
        self.frame += 1

############## IN THIS PART, JUST UNCOMMENT ONE BLOCK #################

#######YOU CAN CHOOSE 4 options :

##### RANDOM VS RANDOM // MCTS vs RANDOM // MCTS vs MCTS or MCTS_training (I used it for training) // MCTS vs MCTS_ANN

# random vs random

# rewards = []
# for i in range(1000):
#     state = CheckerEnv()
#     player=state.active
#     cur_player=WHITE
#     done = False
#     total_reward = 0
#     turn=0
#
#     while not done:
#         action = random.choice(state.available_states()[0])
#         state, reward, done, cur_player = state.step(action)
#         total_reward += reward
#         if not state.jump:  # if not in jump session add a turn
#             turn += 1
#     rewards.append(total_reward)
#     print("finish in {} turns".format(turn))
#     print("Total reward for the game:",total_reward)
#
# print('Mean Reward over 10 episodes:', sum(rewards)/len(rewards))





# mcts (or mcts_play_ANN) vs random

# rewards = []
# for i in range(1):
#     state = CheckerEnv()
#     player=state.active
#     print(player==BLACK)
#     cur_player=BLACK
#     done = False
#     total_reward=0.0
#     turn=0
#     value_net = ValueNetwork()
#     value_net.load_state_dict(torch.load('mon_modele.pth'))
#     value_net.to(device)
#     while not done:
#         if cur_player == player:
#             action, root = mcts_play_ANN(deepcopy(state), value_net)
#         else:
#             action = random.choice(state.available_states()[0])
#         state, reward, done, cur_player = state.step(action)
#         total_reward += reward
#         if not state.jump:  # if not in jump session add a turn
#             turn += 1
#
#     print("finish in {} turns".format(turn))
#     print("Total reward for the game:",total_reward)
#     rewards.append(total_reward)
#
# print('Mean Reward over 10 episodes:', sum(rewards)/len(rewards))


###### uncomment the following block to train your MCTS model (with mcts_training_ANN function)
###### you can also test the result of mcts vs mcts



# mcts vs mcts

rewards = []
value_net = ValueNetwork()
value_net=value_net.create_ANN()

########## print evolution of the loss, useful only if using mcts_training version

app = QApplication(sys.argv)
main_window = RealTimePlot()
main_window.show()

##########

# value_net.load_state_dict(torch.load('mon_modele.pth')) # if commented, you train a new model, else just load a previous model to continue the training

value_net.to(device)
for i in range(1000):
    state = CheckerEnv()
    done = False
    turn=0
    total_reward = 0
    while not done:
        action, root,point = mcts_training_ANN(deepcopy(state), value_net) # you can choose mcts, mcts_training or mcts_ANN
        main_window.new_point=point
        app.processEvents()
        state, reward, done,_ = state.step(action)
        total_reward += reward
        if not state.jump:
            turn += 1
    print("game n° :",i)
    print("finish in {} turns".format(turn))
    rewards.append(total_reward)

print('Mean Reward over 10 episodes:', sum(rewards)/len(rewards))


###### uncomment the following block to test your model_ANN vs the classic mcts : black are the ANN version so if reward ###### is >0, your model is better than the classic mcts


# mcts vs mcts_ANN


# rewards = []
# sum_time_ANN=0
# sum_time_mcts=0
#
# for i in range(10):
#     state = CheckerEnv()
#     player=state.active
#     print(player==BLACK)
#     cur_player=BLACK
#     done = False
#     total_reward=0.0
#     turn=0
#     value_net = ValueNetwork()
#     value_net.load_state_dict(torch.load('mon_modele.pth'))
#     value_net.to(device)
#     while not done:
#         if cur_player == player:
#             time1=time.time()
#             action, root = mcts_play_ANN(deepcopy(state), value_net)
#             sum_time_ANN+=time.time()-time1
#         else:
#             time1=time.time()
#             action,_ = mcts(deepcopy(state))
#             sum_time_mcts+=time.time()-time1
#         state, reward, done, cur_player = state.step(action)
#         total_reward += reward
#         if not state.jump:  # if not in jump session add a turn
#             turn += 1
#     print("game n° :",i)
#     print("finish in {} turns".format(turn))
#     print("Total time for 1 game with mcts :",sum_time_mcts)
#     print("Total time for 1 game with ANN :",sum_time_ANN)
#     print("Total reward for 1 game:",total_reward)
#     rewards.append(total_reward)
# print("Total time for 10 games with mcts :",sum_time_mcts)
# print("Total time for 10 games with ANN :",sum_time_ANN)
# print('Mean Reward over 10 episodes:', sum(rewards)/len(rewards))
