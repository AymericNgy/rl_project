# rl_project

RL project

checker_env.py: environment for the checker game (Monte Carlo version)


--- directory MC version ---
contains the Monte Carlo version of the project

agent_mc.py: class of the agent for the Monte Carlo version of the project 
has a main part to train the agent

evaluate.py: contain functions to evaluate a policy and to plot the results
has a main part to evaluate the agent

get_batch.py: contain functions to generate a batch of episodes

mcts_essai2.py: class contain the Monte Carlo Tree Search algorithm (not usable with the agent_mc.py)
has a main part to test the algorithm

mcts.py: class encapsulate the Monte Carlo Tree Search class (usable with the agent_mc.py)

multi_agent.py: class of MC version with randomly selected policy (from a list policy) to play
has a main part to train the MC agent with an evolving multiAgent instance

plateau_essai2.py: class of the checker game for MCTS 
has a function named copyEnvMCtoMCTS to copy an environment from the MC version to the MCTS version

play_party.py: contain functions to play a game with a policy

utils.py: contain decorator to measure the execution time of a function
