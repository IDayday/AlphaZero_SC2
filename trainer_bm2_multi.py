import os
from socket import timeout
import numpy as np
import random
import time
import torch
import torch.optim as optim
from cmath import *
from wrappers import to_tensor, obs2tensor, cal_steps
from monte_carlo_tree_search_bm import MCTS
from copy import deepcopy
from bm_agent import BMAgent, BMAgent_E
from bm_model import Agent


def simulation(rank, env, model_param, args, device, data_queue, signal_queue, simlog_queue):
    # print('parent process:', os.getppid())
    print('process id:', os.getpid())
    print(f'rank {rank} simulation start')
    obs = env.reset()
    model = Agent(args.n_state, args.hidden_size, args.n_action, device, lr=args.lr, batch_size=args.batch_size,
                    memory_size=args.memory_size, gamma=args.gamma, clip_grad=args.clip_grad)
    model.act_net.load_state_dict(model_param.state_dict())
    agent = BMAgent(model, device, args)
    seed = random.randint(0,1000)
    torch.manual_seed(seed)
    # the np.random.seed will fork the main process, so we need reset again
    np.random.seed(seed)
    temp = 1
    max_search = args.num_simulations
    
    while True:
        train_examples = []
        current_player = 1 # our first
        gameover = False
        obs = env.reset()
        wait_update = False
        while signal_queue.qsize() > 0:
            time.sleep(0.02)
            wait_update = True
        if wait_update:
            model.act_net.load_state_dict(model_param.state_dict())
        sim = time.time()
        while not gameover:
            # decay temperature parameter
            _t = (args.numIters - temp)/args.numIters
            t = _t if _t >= 0.015 else 0.015
            # simulating to get a root, and use it to take one step
            mcts = MCTS(env, obs, agent, args, 'cpu')
            root, root_action_probs, root_available_actions = mcts.run(current_player, max_search)
            root_action_probs_dummy = torch.zeros((1,5))
            for a,q in enumerate(root_action_probs):
                root_action_probs_dummy[0][root_available_actions[a]] = q
            root_action_probs = root_action_probs_dummy
            action_probs = [0 for _ in range(5)]
            for k, v in root.children.items():
                action_probs[k] = v.visit_count
            action_probs = torch.tensor(action_probs / np.sum(action_probs)).float().reshape(1,-1)
            # take a step and update obs and player
            # if temperature = 0, means argmax, but we need random
            # first we warm_up use random policy, then we select a action from the action_prob
            # if epoch < args.warm_up:
            #     action = root.select_action(temperature=inf)
            # else:
            #     action = root.select_action(temperature=t)
            action = root.select_action(temperature=t)
            obs, _ = env.step(obs, action, current_player)
            next_state = obs2tensor(obs, current_player)
            train_examples.append((root.state, current_player, action, action_probs, next_state))

            gameover = env.check_gameover(obs)
            # the value is from the parent's perspective, but it store in the child node
            # so, if we check the value in a node, remember that it reflects your opponent's view
            value = env.get_value(obs, current_player) if gameover else None
            current_player *= -1
            max_search = cal_steps(root_action_probs, action_probs, args.num_simulations)

            # until game is over, we get the value of the game
            if value is not None:
                ret = []
                for hist_state, hist_current_player, hist_action, hist_action_probs, hist_next_state in train_examples:
                    # (state, actionProbabilities, value)
                    # reward = value * ((-1) ** (hist_current_player == current_player))*2/len(train_examples)
                    reward = value * ((-1) ** (hist_current_player == current_player))
                    ret.append((hist_state, hist_action, hist_action_probs, reward, hist_next_state))

                # pipe.send([ret, obs])
                # return ret, obs
                data_queue.put(ret)
                sim_end = round(time.time() - sim,2)
                sim_log = [rank, temp, obs, sim_end]
                simlog_queue.put(sim_log)

                temp += 1


def learn(model_param, args, device, data_queue, signal_queue, simlog_queue):
    # set for numpy and torch
    # os.environ['OPENBLAS_NUM_THREADS'] = '1'
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    # torch.set_num_threads(1)
    model = Agent(args.n_state, args.hidden_size, args.n_action, device, lr=args.lr, batch_size=args.batch_size,
                    memory_size=args.memory_size, gamma=args.gamma, clip_grad=args.clip_grad)
    model.act_net.load_state_dict(model_param.state_dict())
    start = time.time()
    info_log = []

    for i in range(1, args.numIters + 1):
        print("simulation start : {}/{}".format(i, args.numIters))
        info_log.append("simulation start : {}/{} \n".format(i, args.numIters))
        # simulation of the whole game
        for p in range(args.num_processes):
            sim_log = simlog_queue.get()
            rank, temp, obs_info, sim_end = sim_log
            print(f"rank {rank} simulation {temp} cost {sim_end} second")
            info_log.append(f"rank {rank} simulation {temp} cost {sim_end} second \n")
            print(f"simulation result : {obs_info} \n")
            info_log.append(f"simulation result : {obs_info} \n")
            del sim_log

        if data_queue.qsize()%args.num_processes == 0:
            signal_queue.put(1)
            for n in range(args.num_processes):
                ret = data_queue.get()
                ret_clone = deepcopy(ret)
                model.cache.extend(ret_clone)
                del ret
                del ret_clone

            # if the data is enough, start training and update model
            if len(model.cache) > args.batch_size:
                print('start training!')
                info_log.append('start training! \n')
                for j in range(args.epochs):
                    loss = model.update_act()
                    print(f"epoch {j} batchloss: {loss}")
                    info_log.append(f"epoch {j} batchloss: {loss} \n")
                # update model parameter
                model_param.load_state_dict(model.act_net.state_dict())
                if i % args.checkpoint_iter == 0:
                    model.save(args.model_path, i)
                    print("model saved")
                    info_log.append("model saved \n")
                f = open(args.model_path + "log.txt", mode='a')
                for info in info_log:
                    f.write(info)
                f.close()
                info_log = []
                _ = signal_queue.get()
        
        else:
            time.sleep(0.1)     

    end = round((time.time() - start)/60,2)
    print(f"learning completed! cost time: {end} min")
    info_log.append(f"learning completed! cost time: {end} min \n")

    f = open(args.model_path + "log.txt", mode='a')
    for info in info_log:
        f.write(info)
    f.close()


def evaluate(env, model_param, signal_queue, args):
    time.sleep(500)
    while True:
        model_list = os.listdir(args.model_path)
        model_list.remove('log.txt')
        if len(model_list) > 1:
            print("Evaluating!")
            my_model = Agent(args.n_state, args.hidden_size, args.n_action, 'cpu')
            # my_model.act_net.load_state_dict(model_param.state_dict())
            my_model.load(os.path.join(args.model_path,model_list[-1]))
            my_agent = BMAgent_E(my_model, 'cpu', args)
            op_model = Agent(args.n_state, args.hidden_size, args.n_action, 'cpu')
            # beat with all of historial policies
            for k in model_list[:-1]:
                op_model.load(os.path.join(args.model_path, k))
                op_agent = BMAgent_E(op_model, 'cpu', args)
                obs = env.reset()
                player = 1
                gameover = False
                my_win = 0
                for t in range(100):
                    if t <50:
                        while not gameover:
                            if player == 1:
                                action_prob, _, _, available_actions = my_agent.predict(obs, player)
                            elif player == -1:
                                action_prob, _, _, available_actions = op_agent.predict(obs, player)
                            p = np.array(action_prob)
                            p /= p.sum()
                            # max_prob = action_prob.argmax(-1)
                            # action = available_actions[max_prob]
                            action = np.random.choice(available_actions, p=p)
                            obs, _ = env.step(obs, action, player)
                            gameover = env.check_gameover(obs)
                            player *= -1
                        value = env.get_value(obs, player)
                        if value > 0:
                            if player == 1:
                                my_win += 1
                        elif value == 0:
                            my_win += 0
                        elif value < 0:
                            if player == -1:
                                my_win += 1
                    else:
                        while not gameover:
                            if player == 1:
                                action_prob, _, _, available_actions = op_agent.predict(obs, player)
                            elif player == -1:
                                action_prob, _, _, available_actions = my_agent.predict(obs, player)
                            p = np.array(action_prob)
                            p /= p.sum()
                            # max_prob = action_prob.argmax(-1)
                            # action = available_actions[max_prob]
                            action = np.random.choice(available_actions, p=p)
                            obs, _ = env.step(obs, action, player)
                            gameover = env.check_gameover(obs)
                            player *= -1
                        value = env.get_value(obs, player)
                        if value > 0:
                            if player == -1:
                                my_win += 1
                        elif value == 0:
                            my_win += 0
                        elif value < 0:
                            if player == 1:
                                my_win += 1
                my_win_rate = my_win/100
                print(f"{model_list[-1]} vs {k} win rate is {my_win_rate}")
        time.sleep(600)
                