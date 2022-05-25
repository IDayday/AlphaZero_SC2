import os
import numpy as np
import random
import time
import torch
import torch.optim as optim
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

        train_examples = []
        current_player = 1 # our first
        # decay temperature parameter
        t = (self.args.numIters - epoch)/self.args.numIters
        
        while True:
            # simulating to get a root, and use it to take one step
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

                return ret, obs

    def learn(self):
        start = time.time()
        info_log = []
        for i in range(1, self.args.numIters + 1):

            print("simulation start : {}/{}".format(i, self.args.numIters))
            info_log.append("simulation start : {}/{} \n".format(i, self.args.numIters))
            # copying env for simulation
            obs_copy = deepcopy(self.env.reset())
            # simulation of the whole game
            sim = time.time()
            ret, obs_info = self.simulation(obs_copy, i)
            print(f"simulation result : {obs_info}")
            info_log.append(f"simulation result : {obs_info} \n")
            sim_end = round(time.time() - sim,2)
            print(f"simulation {i} cost {sim_end} second")
            info_log.append(f"simulation {i} cost {sim_end} second \n")
            self.agent.model.cache.extend(ret)
            print('simulation is over!')
            # if the data is enough, start training and update model
            if len(self.agent.model.cache) > self.args.batch_size and i >= self.args.warm_up:
                print('start training!')
                info_log.append('start training! \n')
                for j in range(self.args.epochs):
                    loss = self.agent.model.update_act()
                    print(f"epoch {j} batchloss: {loss}")
                    info_log.append(f"epoch {j} batchloss: {loss} \n")
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
                            player, reward = self.evaluate(my_agent, op_agent)
                            print(f"winner is {player} , reward is {reward} \n")
                            info_log.append(f"winner is {player} , reward is {reward} \n")

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
            # use maxprob action
            # max_prob = action_prob.argmax(-1)
            # action = available_actions[max_prob]
            # use sample
            action = np.random.choice(available_actions, p=action_prob)
            obs, _ = self.env.step(obs, action, player)
            gameover = self.env.check_gameover(obs)
            player *= -1
        value = self.env.get_value(obs, player)
        if value > 0:
            return player, obs[8]
        elif value == 0:
            return 0, obs[8]
        elif value < 0:
            return -1*player, obs[8]
            