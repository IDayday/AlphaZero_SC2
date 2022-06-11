from bmgame import BMGame
from bm_agent import DQNBMAgent, DQNBMAgent_E
from bm_model import DQNAgent
import os 
import random
import numpy as np
import time
import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def evaluate(agent1, agent2, flag):
    env = BMGame()
    obs = env.reset()
    player = 1
    gameover = False
    prob_list = []
    while not gameover:
        if player == 1:
            action_prob, _, _, available_actions = agent1.predict(obs, player)
        elif player == -1:
            action_prob, _, _, available_actions = agent2.predict(obs, player)
        p = np.array(action_prob)
        p /= p.sum()
        if flag == 1:
            if player == 1:
                prob_list.append(p)
        elif flag == 2:
            if player == -1:
                prob_list.append(p)
        if (player == 1 and flag == 1) or (player == -1 and flag == 2):
            action = np.random.choice(available_actions, p=p)
        else:
            max_prob = action_prob.argmax(-1)
            action = available_actions[max_prob]
        # use maxprob action
        # max_prob = action_prob.argmax(-1)
        # action = available_actions[max_prob]
        # use sample
        # action = np.random.choice(available_actions, p=p)
        obs, _ = env.step(obs, action, player)
        gameover = env.check_gameover(obs)
        player *= -1
    value = env.get_value(obs, player)
    # print(obs)
    if value > 0:
        return player, prob_list
    elif value == 0:
        return 0, prob_list
    elif value < 0:
        return -1*player, prob_list

def cal_entropy(prob):
    res = []
    for e in prob:
        e = np.array(e)
        entropy = -(e*np.log2(e)).sum()
        res.append(entropy)
    return res

def childprocess(my_model_name, model_list):
    my_model_path = './checkpoint/default_evaluate_sample/'
    model_path = './checkpoint/default_evaluate_max/'   
    my_model = DQNAgent(18, 256, 5, 'cpu')
    my_model.load(os.path.join(my_model_path, my_model_name))
    print('my model', my_model_name)
    my_agent = DQNBMAgent_E(my_model, 'cpu')
    op_model = DQNAgent(18, 256, 5, 'cpu')
    win_log = {}
    entropy_list = []
    start_time = time.time()
    for m in model_list:
        op_model.load(os.path.join(model_path, m))
        op_agent = DQNBMAgent_E(op_model, 'cpu')
        win_rate = 0
        win_times = 0
        # print("op model", m)
        for n in range(50):
            winner, prob_list = evaluate(my_agent, op_agent, 1)
            entropy = cal_entropy(prob_list)
            entropy_list.extend(entropy)
            # total_entropy += entropy
            if winner == 1:
                win_times += 1
        # change player
        for l in range(50):
            winner, prob_list = evaluate(op_agent, my_agent, 2)
            entropy = cal_entropy(prob_list)
            entropy_list.extend(entropy)
            # total_entropy += entropy
            if winner == -1:
                win_times += 1
        win_rate = win_times / 100
        win_log[my_model_name + "_vs_" + m] = win_rate
    entropy_mean = np.mean(np.array(entropy_list))
    entropy_std = np.std(np.array(entropy_list))
    entropy_data = np.vstack((entropy_mean,entropy_std))
    cost_time = time.time() - start_time
    print("time cost", cost_time)
    return [win_log, cost_time, entropy_data]


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=46)
    my_model_path = './checkpoint/default_evaluate_sample/'
    model_path = './checkpoint/default_evaluate_max/'
    model_list = os.listdir(model_path)
    model_list.sort()
    my_model_list = os.listdir(my_model_path)
    my_model_list.sort()
    result = []
    entropy_log = []
    main_start_time = time.time()
    for m in my_model_list:
        result.append(pool.apply_async(childprocess, (m, model_list)))
    pool.close()
    pool.join()
    end = time.time() - main_start_time
    print("main process time cost", end)
    with open("./evaluatelog_sample_vs_max.txt", mode='a') as f:
        for res in result:
            win_log, cost_time, entropy = res.get()
            entropy_log.append(entropy)
            f.write(str(win_log)+'\n')
        entropy_data_smooth = gaussian_filter1d(np.array(entropy_log).squeeze()[:,0], sigma=2)
        error_entropy_data_smooth = gaussian_filter1d(np.array(entropy_log).squeeze()[:,1], sigma=2)
        num = len(error_entropy_data_smooth)
        x_axis = np.linspace(0, num, num)
        plt.figure(figsize=(12,8))
        ax1=plt.subplot(111)
        plt.title('entropy',fontsize=20)
        ax1.plot(entropy_data_smooth, color = 'blue', linewidth = 1.00)
        ax1.fill_between(x_axis, entropy_data_smooth-error_entropy_data_smooth, entropy_data_smooth+error_entropy_data_smooth, facecolor='blue', edgecolor='blue', alpha=0.15)
        plt.tick_params(labelsize=20)
        plt.savefig('./entropy_sample_vs_max.jpg', dpi=200)

    