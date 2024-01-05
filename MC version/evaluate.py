from checker_env import CheckerEnv
from numpy import random
from tqdm import tqdm
from agent_mc import Policy

if __name__ == '__main__':

    env = CheckerEnv()

    rewards = []

    model = Policy()
    model.load()

    for i in range(10):
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
            state, reward, done, cur_player = env.step(action)
            total_reward += reward
            if not env.jump:  # if not in jump session add a turn
                turn += 1
        print("finish in {} turns".format(turn))
        rewards.append(total_reward)

    print('Mean Reward :', sum(rewards) / len(rewards))
