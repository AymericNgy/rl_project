from agent_mc import Policy
from numpy import random
import matplotlib.pyplot as plt
import torch.nn as nn
from worker import get_batch
import torch.optim as optim
import evaluate
import copy
import numpy as np


class MultiAgent(Policy):
    """
    MC version with randomly selected policy (from list_policy) to play
    no minimax evaluation
    """

    def __init__(self, list_policy):
        """
        :param list_policy: list of policy to play with
        """
        self.list_policy = list_policy
        self.lambda_param = 0.5  # probability distribution to sample policies
        self.distribution = np.array([1])

    def add_policy(self, policy):
        """
        add a policy to list_policy
        """
        self.list_policy.append(policy)

        # update probability distribution (exponential decreasing distribution)
        weight = [self.lambda_param ** i for i in range(len(self.list_policy))]
        weight = np.array(weight)
        self.distribution = weight / np.sum(weight)

    def get_move_to_act(self, env):
        index_choisi = np.random.choice(len(self.distribution), size=1, p=self.distribution)[0]
        policy = self.list_policy[index_choisi]

        return policy.get_move_to_act(env)


class TunedPolicy(Policy):
    """
    Policy with a different train method :
    add itself to multi-agent nemesis each time it beat the nemesis with a win rate > threshold
    """

    def __init__(self, minimax_evaluation=False):
        super().__init__(minimax_evaluation)

    def train(self, learning_rate=0.01, plot=False,
              nemesis_model=None):
        """
        train the model,
        add itself to multi-agent nemesis each time it beat the nemesis with a win rate > threshold
        model_name_save : name of the model saved
        num_epochs : number of epochs
        learning_rate : learning rate
        number_of_parties_for_batch : number of parties for each batch
        nemesis_model : model to play against, NEED TO BE INSTANCE OF MULTI-AGENT
        """
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        if nemesis_model:
            nemesis_model.is_an_opponent = True

        # metrics
        losses = []
        win_means = []
        lose_means = []
        draw_means = []
        epochs = []

        win_rate_treshold = 0.9

        iteration = 1000

        for i in range(iteration):
            print(f"--- Iteration {i} ---")
            win_rate = 0
            epoch = 0
            while win_rate < win_rate_treshold:

                # generate data
                number_of_parties_for_batch = 1

                states, rewards = get_batch(number_of_parties_for_batch, self, nemesis_model=nemesis_model)

                values = self.evaluate_value(states)
                loss = criterion(values.squeeze(), rewards)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"Epoch {epoch}")

                if epoch % 2 == 0:
                    loss = loss.item()
                    losses.append(loss)
                    parties_lose_mean, parties_win_mean, parties_draw_mean = evaluate.evaluate_policy(self,
                                                                                                      number_of_parties=10,
                                                                                                      nemesis=nemesis_model,
                                                                                                      verbose=False)
                    print(
                        f'Epoch [{epoch + 1}], Loss: {loss:.4f}, Win: {parties_win_mean:.4f}, Draw : {parties_draw_mean}, Lose : {parties_lose_mean}')
                    win_means += [parties_win_mean]
                    lose_means += [parties_lose_mean]
                    draw_means += [parties_draw_mean]
                    epochs.append(epoch)
                    win_rate = parties_win_mean
                epoch += 1
            save_file_name = f"model_pull/model_{i}.pt"
            self.save_absolute(save_file_name)
            nemesis_model.add_policy(copy.deepcopy(self))
            print(f"{epoch} epochs has been used to beat the multi agent model")


if __name__ == '__main__':
    import checker_env

    # --- TO MODIFY ---

    # --- nemesis model ---
    first_nemesis_model = Policy()
    first_nemesis_model.load_absolute("model_pull/model_80p.pt")
    first_nemesis_model.color_of_model = checker_env.WHITE  # [!] depending if model begin
    first_nemesis_model.is_an_opponent = True

    nemesis_model = MultiAgent([first_nemesis_model])

    # --- policy to train ---
    policy_to_train = TunedPolicy(minimax_evaluation=False)
    policy_to_train.load_absolute("model_pull/model_80p.pt")

    # --- END TO MODIFY ---

    policy_to_train.train(nemesis_model=nemesis_model)
