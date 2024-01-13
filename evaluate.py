from checker_env import CheckerEnv
from numpy import random

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
import time


def play_party(policy, env, first_turn_of_model, color_of_model, show_env=False, verbose=False, nemesis=None):
    state, cur_player = env.reset()
    done = False
    turn = 0

    if show_env:
        print(env)

    while not done:
        moves, states = env.available_states()

        # choose action
        if turn % 2 == first_turn_of_model:
            action = policy.get_move_to_act(env)
        else:
            if nemesis == None:
                action = random.choice(moves)
            else:
                action = nemesis.get_move_to_act(env)
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
        return "party draw"
    elif color_looser_player == color_of_model:
        return "party lose"
    else:
        return "party win"


def evaluate_policy(policy, number_of_parties, show_env=False, verbose=False, nemesis=None):
    # print(policy is nemesis)
    env = CheckerEnv()

    first_turn_of_model = 0  # [!] use it many times in code need to be global or other
    color_of_model = first_turn_of_model
    parties_win = 0
    parties_lose = 0
    parties_draw = 0

    for party in tqdm(range(number_of_parties)):
        res = play_party(policy, env, first_turn_of_model, color_of_model, show_env, verbose, nemesis)
        if res == "party draw":
            parties_draw += 1
        if res == "party lose":
            parties_lose += 1
        if res == "party win":
            parties_win += 1

    parties_lose_mean = parties_lose / number_of_parties
    parties_win_mean = parties_win / number_of_parties
    parties_draw_mean = parties_draw / number_of_parties

    return parties_lose_mean, parties_win_mean, parties_draw_mean


# def evaluate_against_random(policy, number_of_parties=100, show_env=False, verbose=False):
#     """
#     use thread
#     """
#     thread_number = 1
#     parties_per_thread = number_of_parties // thread_number
#
#     with ThreadPoolExecutor(max_workers=thread_number) as executor:
#         # Lancez les threads et récupérez les objets futurs
#         future_to_id = {
#             executor.submit(evaluate_against_random_single, policy, parties_per_thread, show_env, verbose): i + 1 for i
#             in range(5)}
#
#         # Attendez que tous les threads se terminent et récupérez les résultats
#         for future, id_thread in future_to_id.items():
#             try:
#                 resultat = future.result()  # Obtenez le résultat du thread
#                 print(f"Résultat du thread {id_thread}: {resultat}")
#             except Exception as e:
#                 print(f"Thread {id_thread} a levé une exception: {e}")


def evaluate_against_random_thread(policy, number_of_parties=500, thread_number=5, show_env=False, verbose=False):
    start_time = time.time()
    parties_per_thread = number_of_parties // thread_number
    print(parties_per_thread)
    with ThreadPoolExecutor(max_workers=thread_number) as executor:
        # Créez des tâches pour évaluer la politique dans différents threads
        futures = [executor.submit(evaluate_policy, policy, parties_per_thread, show_env, verbose) for _
                   in
                   range(thread_number)]

        # Récupérez les résultats de chaque tâche
        results = [future.result() for future in futures]

        # Affichez ou utilisez les résultats comme nécessaire

        metrics = torch.tensor(results)
        metric = metrics.mean(axis=0)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Execution time :", elapsed_time)

        parties_lose_mean, parties_win_mean, parties_draw_mean = metric[0], metric[1], metric[2]

        return parties_lose_mean, parties_win_mean, parties_draw_mean


def display_pie(parties_lose_mean, parties_win_mean, parties_draw_mean):
    # Étiquettes pour les tranches du camembert
    labels = ['Lose', 'Win', 'Draw']

    # Valeurs pour chaque tranche
    sizes = [parties_lose_mean, parties_win_mean, parties_draw_mean]

    # Couleurs pour chaque tranche
    colors = ['red', 'green', 'orange']

    # Séparation d'une tranche (avec une plus grande valeur) pour mettre en évidence
    explode = (0.1, 0, 0)  # Explosion de la 1ère tranche (Lose)

    # Créer le camembert
    plt.figure(figsize=(7, 7))  # Taille du camembert
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

    # Ajouter un titre
    plt.title('Répartition des parties')

    # Afficher le camembert
    plt.axis('equal')  # Assurez-vous que le camembert est circulaire
    plt.show()


if __name__ == '__main__':

    # deal with MC_version module
    mc_version_path = 'MC_version'

    # Ajoutez le chemin d'accès à sys.path
    if mc_version_path not in sys.path:
        sys.path.append(mc_version_path)

    # Maintenant, essayez d'importer le module
    import agent_mc

    # --- TO MODIFY ---

    nemesis_model = agent_mc.Policy(minimax_evaluation=False)
    nemesis_model.load_absolute("MC_version/model_save/model_6.pt")

    policy = agent_mc.Policy(minimax_evaluation=False)
    policy.load_absolute("MC_version/model_save/model_6.pt")

    number_of_parties = 100



    parties_lose_mean, parties_win_mean, parties_draw_mean = evaluate_policy(policy, number_of_parties,
                                                                             nemesis=policy, verbose=True)

    # --- END TO MODIFY ---

    print(f'Win: {parties_win_mean:.4f}, Draw : {parties_draw_mean}, Lose : {parties_lose_mean}')

    display_pie(parties_lose_mean, parties_win_mean, parties_draw_mean)
