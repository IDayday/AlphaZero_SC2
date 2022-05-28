from bmgame import BMGame
from dqn_agent import DQNBMAgent, DQNBMAgent_E
from dqn_model import DQNAgent
import os 
import random
import numpy as np
import time
import multiprocessing

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def evaluate(agent1, agent2):
    env = BMGame()
    obs = env.reset()
    player = 1
    gameover = False
    while not gameover:
        if player == 1:
            action_prob, _, _, available_actions = agent1.predict(obs, player)
        elif player == -1:
            action_prob, _, _, available_actions = agent2.predict(obs, player)
        p = np.array(action_prob)
        p /= p.sum()
        # use maxprob action
        # max_prob = action_prob.argmax(-1)
        # action = available_actions[max_prob]
        # use sample
        action = np.random.choice(available_actions, p=p)
        obs, _ = env.step(obs, action, player)
        gameover = env.check_gameover(obs)
        player *= -1
    value = env.get_value(obs, player)
    # print(obs)
    if value > 0:
        return player
    elif value == 0:
        return 0
    elif value < 0:
        return -1*player


def childprocess(my_model_name, model_list):
    model_path = './checkpoint/default_evaluate_sample/'   
    my_model = DQNAgent(18, 256, 5, 'cpu')
    my_model.load(os.path.join(model_path, my_model_name))
    print('my model', my_model_name)
    my_agent = DQNBMAgent_E(my_model, 'cpu')
    op_model = DQNAgent(18, 256, 5, 'cpu')
    win_log = {}
    start_time = time.time()
    for m in model_list:
        op_model.load(os.path.join(model_path, m))
        op_agent = DQNBMAgent_E(op_model, 'cpu')
        win_rate = 0
        win_times = 0
        # print("op model", m)
        for n in range(50):
            winner = evaluate(my_agent, op_agent)
            if winner == 1:
                win_times += 1
        # change player
        for l in range(50):
            winner = evaluate(op_agent, my_agent)
            if winner == -1:
                win_times += 1
        win_rate = win_times / 100
        win_log[my_model_name + "_vs_" + m] = win_rate
    cost_time = time.time() - start_time
    print("time cost", cost_time)
    return [win_log, cost_time]


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=46)
    model_path = './checkpoint/default_evaluate_sample/'
    model_list = os.listdir(model_path)
    model_list.sort()
    result = []
    main_start_time = time.time()
    for m in model_list:
        result.append(pool.apply_async(childprocess, (m, model_list)))
    pool.close()
    pool.join()
    end = time.time() - main_start_time
    print("main process time cost", end)
    with open("./evaluatelog.txt", mode='a') as f:
        for res in result:
            win_log, cost_time = res.get()
            f.write(str(win_log)+'\n')
    