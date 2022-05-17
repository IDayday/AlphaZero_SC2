import os
import numpy as np
from random import shuffle

import torch
import torch.optim as optim

from wrappers import make_env, to_tensor, check_bug_initiate, env_reset
from monte_carlo_tree_search_bm import MCTS
from copy import deepcopy, copy

from pysc2.lib.actions import FUNCTIONS as F

class Trainer:

    def __init__(self, env_lsit, agent_list, args, device):
        self.envs = env_lsit[0]
        self.envs_copy = env_lsit[1]
        self.agent = agent_list[0]
        self.agent_copy = agent_list[1]
        self.args = args
        self.device = device

    def simulation(self, action_list):
        env_copy = self.envs_copy[0]
        env_mirror_copy = self.envs_copy[1]
        agent_copy = self.agent_copy
        # env and agent initialize
        env_reset(env_copy, env_mirror_copy, agent_copy, self.args)
        train_examples = []
        current_player = 1  # 我方先手
        
        # 当前环境执行action_list中的动作（marine_agent中的动作函数），更新root_node状态，每次从新的root_node进行搜索
        for i, act in enumerate(action_list):
            # 在执行动作时，agent内记录对局情况的参数也会相应改变
            # 我方
            if i%2==0:
                obs = env_copy.step(actions=[act])[0]
            # 对手
            else:
                obs_mirror = env_mirror_copy.step(actions=[act])[0]

        while True:
            # 依据最大模拟数进行一次从根节点的模拟
            # 用copy的环境进行模拟推演，返回root节点
            self.mcts = MCTS(env_copy, env_mirror_copy, agent_copy, self.args)
            root = self.mcts.run(obs, obs_mirror, current_player)

            action_probs = [0 for _ in range(6)]
            for k, v in root.children.items():
                action_probs[k] = v.visit_count

            action_probs = action_probs / np.sum(action_probs)
            train_examples.append((canonical_board, current_player, action_probs))

            action = root.select_action(temperature=0)
            state, current_player = self.env.get_next_state(state, current_player, action)
            reward = self.env.get_reward_for_player(state, current_player)
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

            train_examples = []
            action_list = []
            gameover = False
            # 通过模拟指导完整对局
            while not gameover:
                action, prob, state, gameover = self.simulation(action_list)
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
