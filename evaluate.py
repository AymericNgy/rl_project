from checker_env import CheckerEnv
from numpy import random

import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def evaluate_against_random(policy, number_of_parties=100, show_env=False, verbose=False):
    env = CheckerEnv()

    first_turn_of_model = 0  # [!] use it many times in code need to be global or other
    color_of_model = first_turn_of_model
    parties_win = 0
    parties_lose = 0
    parties_draw = 0

    for party in range(number_of_parties):

        state, cur_player = env.reset()
        done = False
        turn = 0

        if show_env:
            print(env)

        while not done:
            moves, states = env.available_states()

            # choose action
            if turn % 2 == first_turn_of_model:
                index = policy.get_index_to_act(states)
                action = moves[index]
            else:
                action = random.choice(moves)
            state, reward, terminated, truncated, cur_player = env.step(action)
            done = terminated or truncated
            # total_reward += reward

            if not env.jump:  # if not in jump session add a turn
                turn += 1
            if show_env:
                print(env)

        if verbose:
            print(" --- total turn ---", turn)

        color_looser_player = env.active

        if truncated:
            parties_draw += 1
        elif color_looser_player == color_of_model:  # lose
            parties_lose += 1
        else:  # win
            parties_win += 1

    parties_lose_mean = parties_lose / number_of_parties
    parties_win_mean = parties_win / number_of_parties
    parties_draw_mean = parties_draw / number_of_parties

    return parties_lose_mean, parties_win_mean, parties_draw_mean


if __name__ == '__main__':

    # deal with MC_version module
    mc_version_path = 'MC_version'

    # Ajoutez le chemin d'accès à sys.path
    if mc_version_path not in sys.path:
        sys.path.append(mc_version_path)

    # Maintenant, essayez d'importer le module
    import agent_mc

    policy = agent_mc.Policy()
    policy.load_absolute("MC_version/model_save/model_6.pt")

    number_of_parties = 100
    parties_lose_mean, parties_win_mean, parties_draw_mean = evaluate_against_random(policy, number_of_parties)

    # Définir les catégories et les moyennes correspondantes
    categories = ['Loose', 'Win', 'Draw']
    values = [parties_lose_mean*number_of_parties, parties_win_mean*number_of_parties, parties_draw_mean*number_of_parties]

    # Couleurs correspondantes pour chaque catégorie
    colors = ['red', 'green', 'yellow']

    # Créer le diagramme à barres
    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, values, color=colors)

    # Ajouter des labels et des titres
    plt.xlabel('Categories')
    plt.ylabel('Value')
    plt.title('Values for Different Categories')

    # Ajouter des labels pour chaque barre
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2),
                 va='bottom')  # Affiche la valeur sur chaque barre

    # Afficher le diagramme
    plt.tight_layout()
    plt.show()
