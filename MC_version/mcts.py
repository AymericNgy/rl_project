from agent_mc import Policy
from mcts_essai2 import MCTSNode
from mcts_essai2 import mcts
from plateau_essai2 import copyEnvMCtoMCTS
from copy import deepcopy
from utile import execution_time


class MCTS(Policy):
    """
    use to encapsulate MCTS in Policy class
    """

    def __init__(self):
        super().__init__()

    def evaluate_value(self, state):
        raise RuntimeError("cannot be used with MCTS")

    def minimax_value(self, env, depth=3):
        raise RuntimeError("cannot be used with MCTS")

    def get_move_to_act_from_minimax(self, env):
        raise RuntimeError("cannot be used with MCTS")

    @execution_time
    def get_move_to_act(self, env):
        env_MCTS = copyEnvMCtoMCTS(env)

        action, root = mcts(env_MCTS)
        return action

    def train(self, model_name_save, num_epochs=100, learning_rate=0.01, number_of_parties_for_batch=100, plot=False,
              nemesis_model=None):
        raise RuntimeError("cannot be used with MCTS")

    def save(self, name='model_default_save'):
        raise RuntimeError("cannot be used with MCTS")

    def load(self):
        raise RuntimeError("cannot be used with MCTS")

    def load_absolute(self, name):
        raise RuntimeError("cannot be used with MCTS")
