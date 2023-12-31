from checker_env import CheckerEnv
from numpy import random
from tqdm import tqdm
from agent_mc import Policy

# DEPRECATED


if __name__ == '__main__':
    print("deprecated")

    env = CheckerEnv()

    rewards = []

    model = Policy()
    model.load()

    wins = 0
    loses = 0
    draws = 0

    for i in tqdm(range(100)):
        state, cur_player = env.reset()
        done = False
        total_reward = 0
        turn = 0

        first_turn_of_model = 0  # [!] use it many times in code need to be global or other

        while not done:
            moves, states = env.available_states()

            # choose action
            if turn % 2 == first_turn_of_model:
                index = model.get_index_to_act(states)
                action = moves[index]
            else:
                action = random.choice(moves)

            action = random.choice(moves)
            state, reward, terminated, truncated, cur_player = env.step(action)

            done = terminated or truncated

            total_reward += reward
            if not env.jump:  # if not in jump session add a turn
                turn += 1
        # print("finish in {} turns".format(turn))

        # [!] prendre en compte egalite
        if truncated:
            final_reward = 0.5
            draws += 1
        elif first_turn_of_model == env.active:
            final_reward = 0
            loses += 1
        else:
            final_reward = 1
            wins += 1

        rewards.append(final_reward)

    print(f"wins {wins} loses {loses} draws {draws}")
    print('Mean Reward :', sum(rewards) / len(rewards))
