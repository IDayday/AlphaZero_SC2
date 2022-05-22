import os
import numpy as np
from random import shuffle

import torch
import torch.optim as optim

from wrappers import to_tensor, obs2tensor
from monte_carlo_tree_search_bm import MCTS
from copy import deepcopy, copy


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
            # 依据最大模拟数进行一次从根节点的模拟
            # 用copy的环境进行模拟推演，返回root节点
            self.mcts = MCTS(self.env, obs, self.agent, self.args, self.device)
            root, search_path = self.mcts.run(current_player)

            action_probs = [0 for _ in range(5)]
            for k, v in root.children.items():
                action_probs[k] = v.visit_count

            action_probs = action_probs / np.sum(action_probs)
            train_examples.append((root.state, current_player, action_probs))

            action = root.select_action(temperature=0)
            state, current_player = env.get_next_state(state, current_player, action)
            reward = agent.cal_win_value(root.obs_s)
            # 只有模拟到终局才有reward返回，如果一直没有reward，那么会一直模拟下去
            if reward is not None:
                ret = []
                for hist_state, hist_current_player, hist_action_probs in train_examples:
                    # [Board, currentPlayer, actionProbabilities, Reward]
                    ret.append((hist_state, hist_action_probs, reward * ((-1) ** (hist_current_player != current_player))))

                return ret

    def learn(self):
        for i in range(1, self.args.numIters + 1):

            print("simulation start : {}/{}".format(i, self.args.numIters))
            # copying env for simulation
            obs_copy = deepcopy(self.env.reset())
            train_examples = []
            gameover = False
            # simulation of the whole game
            while not gameover:
                action, prob, state, gameover = self.simulation(obs_copy)
                action_list.append(action)
                trajactory = [action, prob, state]
                train_examples.extend(trajactory)
            print('simulation is over!')
            # 有足够数据时，进行一次训练，然后将新的model参数传递给simulation
            if len(train_examples) > self.args.batch_size:
                print('start training!')
                shuffle(train_examples)
                self.train(train_examples)
                filename = self.args.checkpoint
                self.save_checkpoint(folder=".", filename=filename)

    def train(self, examples):
        optimizer = optim.Adam(self.agent.parameters(), lr=5e-4)
        pi_losses = []
        v_losses = []

        for epoch in range(self.args['epochs']):
            self.agent.train()

            batch_idx = 0

            while batch_idx < int(len(examples) / self.args['batch_size']):
                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                boards = boards.contiguous().to(self.device)
                target_pis = target_pis.contiguous().to(self.device)
                target_vs = target_vs.contiguous().to(self.device)

                # compute output
                out_pi, out_v = self.agent(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.append(float(l_pi))
                v_losses.append(float(l_v))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_idx += 1

            print()
            print("Policy Loss", np.mean(pi_losses))
            print("Value Loss", np.mean(v_losses))
            print("Examples:")
            print(out_pi[0].detach())
            print(target_pis[0])

    def loss_pi(self, targets, outputs):
        loss = -(targets * torch.log(outputs)).sum(dim=1)
        return loss.mean()

    def loss_v(self, targets, outputs):
        loss = torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]
        return loss

    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.agent.state_dict(),
        }, filepath)
