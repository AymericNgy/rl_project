import copy

from checker_env_MC import CheckerEnv
from numpy import random

import matplotlib.pyplot as plt
from tqdm import tqdm


def play_party(policy, env, first_turn_of_model, color_of_model, show_env=False, verbose=False, nemesis=None):
    """
    play a party with policy against random policy or nemesis (if nemesis is not None)
    :param policy: policy to play with
    :param env: environment
    :param first_turn_of_model: 0 if model play first, 1 otherwise
    :param color_of_model: 0 if model is black, 1 otherwise
    :param show_env: show environment
    :param verbose: verbose
    :param nemesis: nemesis to play against (if None : play against random policy)
    """
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
    """
    evaluate policy against random policy or nemesis (if nemesis is not None)
    :param policy: policy to play with
    :param number_of_parties: number of parties to play
    :param show_env: show environment
    :param verbose: verbose
    :param nemesis: nemesis to play against (if None : play against random policy)
    """
    if nemesis:
        nemesis.is_an_opponent = True
        nemesis.color_of_model = 1  # [!] depending on if first

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


def display_pie(parties_lose_mean, parties_win_mean, parties_draw_mean):
    """
    display pie of win, lose and draw
    """

    labels = ['Lose', 'Win', 'Draw']


    sizes = [parties_lose_mean, parties_win_mean, parties_draw_mean]

    colors = ['red', 'green', 'orange']

    explode = (0.1, 0, 0)  # explode 1st slice

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

    plt.title('Répartition des parties')


    plt.axis('equal')  # Assurez-vous que le camembert est circulaire
    plt.show()


if __name__ == '__main__':
    from mcts import MCTS

    # --- TO MODIFY ---

    # --- choose nemesis model---

    # nemesis_model = Policy(minimax_evaluation=False)
    # nemesis_model.load_absolute("model_pull/model_80p.pt")

    nemesis_model = MCTS()

    # --- choose policy to evaluate ---

    # policy = Policy(minimax_evaluation=False, depth_minimax=3)
    # policy.load_absolute("model_pull/model_80p.pt")

    policy = copy.deepcopy(nemesis_model)

    number_of_parties = 10

    parties_lose_mean, parties_win_mean, parties_draw_mean = evaluate_policy(policy, number_of_parties,
                                                                             nemesis=nemesis_model, verbose=True)

    # --- END TO MODIFY ---

    plt.plot(nemesis_model.execution_times)
    plt.yscale('log')

    print(f'Win: {parties_win_mean:.4f}, Draw : {parties_draw_mean}, Lose : {parties_lose_mean}')

    display_pie(parties_lose_mean, parties_win_mean, parties_draw_mean)
