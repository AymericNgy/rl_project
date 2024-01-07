from checker_env import CheckerEnv
from numpy import random
from tqdm import tqdm

if __name__ == '__main__':

    env = CheckerEnv()

    rewards = []
    for i in range(10):
        state, cur_player = env.reset()
        done = False
        total_reward = 0
        turn = 0
        while not done:
            moves, states = env.available_states()
            action = random.choice(moves)
            state, reward, terminated, truncated, cur_player = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if not env.jump:  # if not in jump session add a turn
                turn += 1
        print("finish in {} turns".format(turn))
        rewards.append(total_reward)

    print('Mean Reward over 1000 episodes:', sum(rewards) / len(rewards))
