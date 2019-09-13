import os
import gym
import time
import numpy as np
from parameters import parameters


def preprocess_state(state):
    return np.clip(state, -10, 10).astype('float32')


def process_reward(reward, prev_action, state):
    return 0.01*reward
    # if reward == -100:
    #     return -1.
    # return 0.1 * (state[2] + 0.1 * (np.cos(state[0]) * state[14] - 0.3)) - 0.1 * np.sum(
    #     np.maximum(np.maximum(-prev_action - [1, 1, 1, 1],
    #                           prev_action - [1, 1, 1, 1]), [0, 0, 0, 0]))
    # return 0.1 * reward - 0.1*np.sum(np.maximum(np.maximum(-prev_action - [1, 1, 1, 1],
    #                               prev_action - [1, 1, 1, 1]), [0, 0, 0, 0]))


def generate_game(env, render, pid, process_queue, common_dict):
    observation = env.reset()
    done = False
    reward_list = []
    action_list = []
    observation_list = []
    prob_list = []
    action = np.array([0, 0, 0, 0])
    while not done:
        observation_list.append(observation)
        process_queue.put((pid, observation))
        while pid not in common_dict:
            time.sleep(0.001)
        prev_action = np.copy(action)
        action, prob = common_dict[pid]
        del common_dict[pid]
        observation, reward, done, _ = env.step(np.clip(action, -1, 1))
        action_list.append(action)
        prob_list.append(prob)
        reward_list.append(process_reward(reward, prev_action, observation))
        if render:
            env.render()
    print('Distance: {0:7.3f}'.format(np.sum(observation_list, 0)[2]), flush=True)  # TODO change to distance
    for i in range(len(reward_list) - 2, -1, -1):
        reward_list[i] += reward_list[i + 1] * parameters.GAMMA  # compute the discounted obtained reward for each step
    return observation_list, reward_list, action_list, prob_list


def cpu_thread(render, memory_queue, process_queue, common_dict, worker):
    import psutil
    p = psutil.Process()
    p.cpu_affinity([worker])
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        env = gym.make("BipedalWalker-v2")
        pid = os.getpid()
        print('process started with pid: {} on core {}'.format(os.getpid(), worker), flush=True)
        while True:
            observation_list, reward_list, action_list, prob_list = generate_game(env, render, pid, process_queue, common_dict)
            for i in range(len(observation_list)):
                memory_queue.put((observation_list.pop(), reward_list.pop(), action_list.pop(), prob_list.pop()))
    except Exception as e:
        print(e, flush=True)
