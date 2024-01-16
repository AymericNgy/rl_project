To use these models be sure that :

- you took the valid network : if the model is 16Ko, take the simple network and if is 737Ko, take the complex network in the value_network_checkers.py

- if you want to continue training, uncomment the line # value_net.load_state_dict(torch.load('mon_modele.pth')) in # mcts vs mcts part in mcts_essai2.py

- if you want to test a model, just rename the file mon_modele.pth and put the path in the #value_net.load_state_dict(torch.load('mon_modele.pth')) of mcts vs mcts_ANN part