!!! Be sure to read carefully the comments in the code before launching it or it will not work correctly (because I made several tests, portions of the code are commented)

This file contains 3 .py : 

- mcts_essai2 : contains the MCTS algorithm with the neural network training code
- plateau_essai2 : contains the environement of the checker game and some modified function to interface with the MCTS extracted from the TD of Reinforcement Learning
- value_network_checkers : 2 implementations of value neural network


More in details

In value_network_checkers.py : I implemented 2 different networks, the simple network from an idea of the article "Improving Monte Carlo Tree Search with Artificial Neural Networks without Heuristics" page 11, the second, more complex, is my idea.
See the read_me.txt in the file "models" to know which part you should comment or not in this .py.

In the plateau_essai2.py : I modified some functions because the game is different from TicTacToe

In the mcts_essai2 : I mainly used the MCTS code extracted from TD of Reinforcement Learning but with a new idea : to implement a value network to give an estimation of a state. The idea was to improve the speed of the rollout which is responsible of the main part of the time of the algorithm : indeed, this value network would give a prediction of win given a state and, if the prediction is higher than a threshold of confidence, we pass the rollout and consider that the result is the prediction given by the value_network (see article "Improving Monte Carlo Tree Search with Artificial Neural Networks without Heuristics")

In detail : 

- mcts_play_ANN : used to make a previously trained network playing : we can specify 2 different rollouts ( rollout_version_ANN and rollout_version_ANN_2) but the second version is clearly faster. We can specify more or less iterations but time is impacted.

- rollout_version_ANN and rollout_version_ANN_2 : the idea is to either (version_ANN) : give a prediction of the state for each iteration in the rollout and if the prediction is higher than a threshold, return the winner or (version_ANN_2) : give only a prediction at the beginning of the rollout and then do the complete rollout : I have tested the first and the second but the first is very slow

- mcts_training_ANN : used to train the value network : the algorithm will play against itself to improve the ability to correctly give a value for a state : there are 2 possibilities : you can train the network by comparing 1 result of a rollout to the prediction and update the network (so for example 0.63 and 1 after the rollout) OR the second option : use multiple rollouts to update the network more smoothly : by doing multiple rollouts, a state is better evaluated compared with one rollout so the network will not be updated too hardly
FOLLOW THE COMMENT IN THE CODE TO USE ONE OF THE OPTION


- backpropagate and backpropagate_stochastic : this option is important if you decide to use 1 or multiple rollout because, if multiple rollout are chosen, the code is slightly different (we can't update the parent nodes with +-1 anymore but we update with a probability)

- in the part "mcts vs mcts", I train my neural network, you can change the number of games, continue to train a previous model...

- in the part "mcts vs mcts_ANN", I test my neural network predictive model vs a classic mcts : faster but performance decreased

- some others functions more classical like mcts, random vs random part ... directly taken from the TD

- a curve function with pyqtgraph to dynamically print the training loss : the loss is the sum of the abs(prediction-real) so the loss can change if you change the iter value in mcts or if you use multiple rollout
