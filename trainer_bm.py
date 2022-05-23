import os
import numpy as np
import random
import time
import torch
import torch.optim as optim

from wrappers import to_tensor, obs2tensor
from monte_carlo_tree_search_bm import MCTS
from copy import deepcopy
from dqn_agent import DQNBMAgent
from dqn_model import DQNAgent

class Trainer:

    def __init__(self, env, agent, args, device):
        self.env = env
        self.agent = agent
        self.args = args
        self.device = device

    def simulation(self, obs):

        train_examples = []
        current_player = 1 # our first
        
        while True:
            # simulating to get a root, and use it to take one step
            self.mcts = MCTS(self.env, obs, self.agent, self.args, self.device)
            root = self.mcts.run(current_player)

            action_probs = [0 for _ in range(5)]
            for k, v in root.children.items():
                action_probs[k] = v.visit_count

            action_probs = action_probs / np.sum(action_probs)
            # take a step and update obs and player
            action = root.select_action(temperature=0)
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
                    reward = value * ((-1) ** (hist_current_player == current_player))/len(train_examples)
                    ret.append((hist_state, hist_action, hist_action_probs, reward, hist_next_state))

                return ret

    def learn(self):
        start = time.time()
        for i in range(1, self.args.numIters + 1):

            print("simulation start : {}/{}".format(i, self.args.numIters))
            # copying env for simulation
            obs_copy = deepcopy(self.env.reset())
            # simulation of the whole game
            sim = time.time()
            ret = self.simulation(obs_copy)
            sim_end = round(time.time() - sim,2)
            print(f"simulation {i} cost {sim_end} second")
            self.agent.model.cache.extend(ret)
            print('simulation is over!')
            # if the data is enough, start training and update model
            if len(self.agent.model.cache) > self.args.batch_size:
                print('start training!')
                for j in range(self.args.epochs):
                    loss = self.agent.model.update_act()
                    print(f"epoch {j} batchloss: ",loss)
                # update the target network
                if i % self.args.update_tgt == 0:
                    self.agent.model.update_tgt()
                    print("target network updated")
                if i % self.args.checkpoint_iter == 0:
                    self.agent.model.save(self.args.model_path, i)
                    print("model saved")
                    model_list = os.listdir(self.args.model_path)
                    if len(model_list) >1 :
                        print("Evaluating!")
                        my_model = DQNAgent(18, self.args.hidden_size, 5, 'cpu')
                        my_model.load(os.path.join(self.args.model_path,model_list[-1]))
                        my_agent = DQNBMAgent(my_model, 'cpu')
                        op_model = DQNAgent(18, self.args.hidden_size, 5, 'cpu')
                        op_model.load(os.path.join(self.args.model_path,random.choice(model_list[:-1])))
                        op_agent = DQNBMAgent(op_model, 'cpu')
                        self.evaluate(my_agent, op_agent)
        end = round((time.time() - start)/60,2)
        print(f"learning completed! cost time: {end} min")

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
            print("winner is ",player)
        elif value == 0:
            print("draw")
        elif value < 0:
            print("winner is ",player*-1)
            