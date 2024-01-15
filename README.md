# rl_project

RL project

--- directory MC_version ---

We use reward = 1 for a win, reward = 0.5 for a draw and reward = 0 for a loose.

checker_env_MC.py: environment for the checker game (Monte Carlo version)

contains the Monte Carlo version of the project

agent_mc.py: class of the agent for the Monte Carlo version of the project
has a main part to train the agent

evaluate.py: contain functions to evaluate a policy and to plot the results
has a main part to evaluate the agent

get_batch.py: contain functions to generate a batch of episodes (reward : win:1 draw:0.5 lose:0)

mcts_essai2.py: class contain the Monte Carlo Tree Search algorithm (not usable with the agent_mc.py)
has a main part to test the algorithm

mcts.py: class encapsulate the Monte Carlo Tree Search class (usable with the agent_mc.py)

multi_agent.py: class of MC version with randomly selected policy (from a list policy) to play
has a main part to train the MC agent with an evolving multiAgent instance

plateau_essai2.py: class of the checker game for MCTS
has a function named copyEnvMCtoMCTS to copy an environment from the MC version to the MCTS version

play_party.py: contain functions to play a game with a policy

utils.py: contain decorator to measure the execution time of a function

directory model_save: contains models created by a Policy by default

directory model_pull: contains model created by multi_agent.py
