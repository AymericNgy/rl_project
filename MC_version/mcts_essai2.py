from IPython.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

from copy import deepcopy
import random
from plateau_essai2 import *
import math

# je conserve le truc de aymeric pareil par contre je modif le code : faut juste se dire que le plateau entier (le state) est un instance de checker_env et donc contient tout (player, jump ...)
# class TicTacToe:

random_iters = [30, 35, 40, 45, 50, 55, 60, 65, 70]

random_iters = [50]


# random_iters = None


class MCTSNode:
    """MCTS Node."""

    def __init__(self, state, parent=None, action=None):
        """Initialize a node."""

        self.state = state  # state of the board, a checker_env element
        self.parent = parent
        self.action = action
        self.children = []
        self.untried_actions, _ = self.state.available_states()
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

    def rollout(self):
        """Until termination, move randomly. Return the result (winning player)"""

        state = self.state
        done, result = state.check_termination()
        while not done:
            possible_actions, _ = state.available_states()
            action = random.choice(possible_actions)
            state = state.peek_move(action)
            done, result = state.check_termination()
        return result

    def backpropagate(self, result):
        """Backprop the result of a rollout up to the root node: For each node in the path update the visits and the number of wins"""

        self.n += 1
        if self.parent:
            if result == "White" and self.parent.state.active == WHITE or result == "Black" and self.parent.state.active == BLACK:  # active == BLACK or WHITE
                self.w += 1
            else:
                self.w -= 1
            self.parent.backpropagate(result)

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

        return self.w / self.n

    def uct(self):
        """UCT value of a node"""

        return self.win_ratio() + math.sqrt(2 * math.log(self.parent.n) / self.n)

    def best_child(self):
        """Return the best child (the one with the highest win ratio)"""

        best_win_ratio, child = max((self.children[i].win_ratio(), i) for i in range(len(self.children)))

        return self.children[child]

    def best_uct_child(self):
        """Return the best child according to UCT"""

        best_win_ratio, child = max((self.children[i].uct(), i) for i in range(len(self.children)))

        return self.children[child]


def mcts(state, iters=10):
    if random_iters:
        iters = np.random.choice(random_iters)


    root = MCTSNode(deepcopy(state))

    for i in range(iters):
        leaf = root.traverse()
        simulation_result = leaf.rollout()
        leaf.backpropagate(simulation_result)

    return root.best_child().action, root


if __name__ == '__main__':

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

    # mcts vs random

    rewards = []
    for i in range(10):
        state = CheckerEnvForMCTS()
        player = state.active
        print(player == BLACK)
        cur_player = BLACK
        done = False
        total_reward = 0.0
        turn = 0
        while not done:
            if cur_player == player:
                action, root = mcts(deepcopy(state))
            else:
                action = random.choice(state.available_states()[0])
            state, reward, done, cur_player = state.step(action)
            total_reward += reward
            if not state.jump:  # if not in jump session add a turn
                turn += 1

        print("finish in {} turns".format(turn))
        print("Total reward for the game:", total_reward)
        rewards.append(total_reward)

    print('Mean Reward over 10 episodes:', sum(rewards) / len(rewards))

    # mcts vs mcts

    # rewards = []
    # for i in range(10):
    #     state = CheckerEnv()
    #     # env.reset()
    #     done = False
    #     turn=0
    #     total_reward = 0
    #     while not done:
    #         action, root = mcts(deepcopy(state))
    #         state, reward, done,_ = state.step(action)
    #         total_reward += reward
    #         if not state.jump:  # if not in jump session add a turn
    #             turn += 1
    #     print("finish in {} turns".format(turn))
    #     rewards.append(total_reward)
    #
    # print('Mean Reward over 10 episodes:', sum(rewards)/len(rewards))

    # mettre un minmax dans le rollout
    # test contre aymeric
    # meilleure interface graphique
    # réseau neurone
    # def __init__(self, nemesis_model=None, minimax_evaluation=True):
