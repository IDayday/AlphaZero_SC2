import os
from socket import timeout
import numpy as np
import random
import time
import torch
import torch.optim as optim
import multiprocessing
from cmath import *
from wrappers import to_tensor, obs2tensor
from monte_carlo_tree_search_bm import MCTS
from copy import deepcopy
from dqn_agent import DQNBMAgent, DQNBMAgent_E
from dqn_model import DQNAgent

class Trainer:

    def __init__(self, env, agent, args, device):
        self.env = env
        self.agent = agent
        self.args = args
        self.device = device


    def simulation(self, obs, epoch):
        print('parent process:', os.getppid())
        print('process id:', os.getpid())
        # obs = pipe.recv()
        print('new obs getting')
        seed = random.randint(0,1000)
        torch.manual_seed(seed)
        # the np.random.seed will fork the main process, so we need reset again
        np.random.seed(seed)
        obs = deepcopy(obs)
        train_examples = []
        current_player = 1 # our first
        # decay temperature parameter
        t = (self.args.numIters - epoch)/self.args.numIters
        
        while True:
            # simulating to get a root, and use it to take one step
            if self.args.num_processes > 1:
                self.mcts = MCTS(self.env, obs, self.agent, self.args, 'cpu')
            else:
                self.mcts = MCTS(self.env, obs, self.agent, self.args, self.device)
            root = self.mcts.run(current_player)

            action_probs = [0 for _ in range(5)]
            for k, v in root.children.items():
                action_probs[k] = v.visit_count

            action_probs = action_probs / np.sum(action_probs)
            # take a step and update obs and player
            # if temperature = 0, means argmax, but we need random
            # first we warm_up use random policy, then we select a action from the action_prob
            if epoch < self.args.warm_up:
                action = root.select_action(temperature=inf)
            else:
                action = root.select_action(temperature=t)
            obs, _ = self.env.step(obs, action, current_player)
            next_state = obs2tensor(obs, current_player)
            train_examples.append((root.state, current_player, action, action_probs, next_state))

            gameover = self.env.check_gameover(obs)
            # the value is from the parent's perspective, but it store in the child node
            # so, if we check the value in a node, remember that it reflects your opponent's view
            value = self.env.get_value(obs, current_player) if gameover else None
            current_player *= -1
            # until game is over, we get the value of the game
            if value is not None:
                ret = []
                for hist_state, hist_current_player, hist_action, hist_action_probs, hist_next_state in train_examples:
                    # (state, actionProbabilities, value)
                    # reward = value * ((-1) ** (hist_current_player == current_player))*2/len(train_examples)
                    reward = value * ((-1) ** (hist_current_player == current_player))
                    ret.append((hist_state, hist_action, hist_action_probs, reward, hist_next_state))

                # pipe.send([ret, obs])
                return ret, obs

    def learn(self):
        # set for numpy and torch
        # os.environ['OPENBLAS_NUM_THREADS'] = '1'
        np.set_printoptions(precision=5)
        np.set_printoptions(suppress=True)
        # torch.set_num_threads(1)

        start = time.time()
        info_log = []
        for i in range(1, self.args.numIters + 1):
            pool = multiprocessing.Pool(processes=self.args.num_processes)
            print("simulation start : {}/{}".format(i, self.args.numIters))
            info_log.append("simulation start : {}/{} \n".format(i, self.args.numIters))
            # copying env for simulation
            obs_copy = deepcopy(self.env.reset())
            # simulation of the whole game
            sim = time.time()
            # this simulation process can be parallelized
            ##### pipe for multiprocesses
            # pipe_dict = dict((rank, (pipe1, pipe2)) for rank in range(self.args.num_processes) for pipe1, pipe2 in (multiprocessing.Pipe(),))
            # child_process_list = []
            # for p in range(self.args.num_processes):
            #     pro = multiprocessing.Process(target=self.simulation, args=(pipe_dict[p][1],i,))
            #     child_process_list.append(pro)
            # [pipe_dict[p][0].send(obs_copy) for p in range(self.args.num_processes)]
            # [p.start() for p in child_process_list]

            # for p in range(self.args.num_processes):
            #     print(f'get trans from {p} process')
            #     trans = pipe_dict[p][0].recv()
            #     ret, obs_info = trans[0], trans[1]
            #     # ret, obs_info = self.simulation(obs_copy, i)
            #     self.agent.model.cache.extend(ret)

            #     print(f"simulation result : {obs_info} \n")
            #     info_log.append(f"simulation result : {obs_info} \n")

            # [p.terminate() for p in child_process_list]        
            # print('stop process')
            # [p.join() for p in child_process_list] 



            ##### pool for multiprocesses
            result = []
            for m in range(self.args.num_processes):
                result.append(pool.apply_async(self.simulation, (obs_copy, i)))
            pool.close()
            pool.join()
            sim_end = round(time.time() - sim,2)
            print(f"simulation {i} cost {sim_end} second")
            info_log.append(f"simulation {i} cost {sim_end} second \n")
            print('simulation is over!')
            for res in result:
                # it will get the error: 
                # multiprocessing.pool.MaybeEncodingError: 
                # Error sending result:   Reason: 'OSError(24, 'Too many open files')'
                # this bug may come from limit on number of file descriptors
                # "ulimit -n" can check the limit on number of file descriptors
                ret, obs_info = res.get() 
                self.agent.model.cache.extend(ret)
                print(f"simulation result : {obs_info} \n")
                info_log.append(f"simulation result : {obs_info} \n")

            
            ##### pool for multiprocesses debug
            # result = []
            # # for m in range(self.args.num_processes):
            # with multiprocessing.Pool(processes=self.args.num_processes) as pool:
            #     result.append(pool.apply_async(self.simulation, (obs_copy, i)))
            #     # pool.close()
            #     # pool.join()
            # for res in result:
            #     ret, obs_info = res.get() 
            #     self.agent.model.cache.extend(ret)
            #     print(f"simulation result : {obs_info} \n")
            #     info_log.append(f"simulation result : {obs_info} \n")
            # sim_end = round(time.time() - sim,2)
            # print(f"simulation {i} cost {sim_end} second")
            # info_log.append(f"simulation {i} cost {sim_end} second \n")
            # print('simulation is over!')


            # if the data is enough, start training and update model
            if len(self.agent.model.cache) > self.args.batch_size and i >= self.args.warm_up:
                print('start training!')
                info_log.append('start training! \n')
                for j in range(self.args.epochs):
                    loss = self.agent.model.update_act()
                    print(f"epoch {j} batchloss: {loss}")
                    info_log.append(f"epoch {j} batchloss: {loss} \n")
                # update the target network
                # if i % self.args.update_tgt == 0:
                #     self.agent.model.update_tgt()
                #     print("target network updated")
                #     info_log.append("target network updated \n")
                if i % self.args.checkpoint_iter == 0:
                    self.agent.model.save(self.args.model_path, i)
                    print("model saved")
                    info_log.append("model saved \n")
                    model_list = os.listdir(self.args.model_path)
                    if len(model_list) >1 :
                        print("Evaluating!")
                        info_log.append("Evaluating! \n")
                        my_model = DQNAgent(18, self.args.hidden_size, 5, 'cpu')
                        my_model.load(os.path.join(self.args.model_path,model_list[-1]))
                        my_agent = DQNBMAgent_E(my_model, 'cpu', self.args)
                        op_model = DQNAgent(18, self.args.hidden_size, 5, 'cpu')
                        # beat with all of historial policies
                        for k in model_list[:-1]:
                            op_model.load(os.path.join(self.args.model_path, k))
                            op_agent = DQNBMAgent_E(op_model, 'cpu', self.args)
                            player, result = self.evaluate(my_agent, op_agent)
                            print(f"winner is {player} , result is {result}")
                            info_log.append(f"winner is {player} , result is {result} \n")
            f = open(self.args.model_path + "log.txt", mode='a')
            for info in info_log:
                f.write(info)
            f.close()
            info_log = []

        end = round((time.time() - start)/60,2)
        print(f"learning completed! cost time: {end} min")
        info_log.append(f"learning completed! cost time: {end} min \n")

        f = open(self.args.model_path + "log.txt", mode='a')
        for info in info_log:
            f.write(info)
        f.close()

    def evaluate(self, agent1, agent2):
        obs = self.env.reset()
        player = 1
        gameover = False
        while not gameover:
            if player == 1:
                action_prob, _, _, available_actions = agent1.predict(obs, player)
            elif player == -1:
                action_prob, _, _, available_actions = agent2.predict(obs, player)
            max_prob = action_prob.argmax(-1)
            action = available_actions[max_prob]
            obs, _ = self.env.step(obs, action, player)
            gameover = self.env.check_gameover(obs)
            player *= -1
        value = self.env.get_value(obs, player)
        if value > 0:
            return player, obs
        elif value == 0:
            return 0, obs
        elif value < 0:
            return -1*player, obs
            