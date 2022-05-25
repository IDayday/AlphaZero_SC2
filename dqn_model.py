import random, math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from collections import deque
from wrappers import to_tensor

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.hidden = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.action = nn.Linear(hidden_size, output_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        # action = F.softmax(self.action(self.dropout(x)),dim=-1)
        # value = torch.tanh(self.value(self.dropout(x)))
        action = F.softmax(self.action(x),dim=-1)
        value = torch.tanh(self.value(x))
        return action , value

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.pos = 0

    def push(self, item):
        if len(self.memory) < self.capacity:
            self.memory.extend(item)
        else:
            self.memory[self.pos] = item
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = zip(*random.sample(self.memory, batch_size))
        return [torch.stack(x, dim=0) for x in batch]

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, n_observe, hidden_size, n_action, device,
                 lr=1e-2, batch_size=64, memory_size=10000, gamma=0.99,
                 clip_grad=1.0, eps_start=0.9, eps_decay=200, eps_end=0.05):
        self.n_observe = n_observe
        self.hidden_size = hidden_size
        self.n_action = n_action

        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.gamma = gamma
        self.clip_grad = clip_grad
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end

        # self.tgt_net = DQN(n_observe, hidden_size, n_action).to(device)
        self.act_net = DQN(n_observe, hidden_size, n_action).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.act_net.parameters(), lr=lr)
        self.cache = deque(maxlen=self.memory_size) # or Memory(maxlen=self.memory_size)
        self.steps_done = 0

        self.act_net.apply(self.initialize_weights)
        # self.update_tgt()
        self.act_net.train()
        # self.tgt_net.eval()

    @staticmethod
    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    # def update_tgt(self):
    #     self.tgt_net.load_state_dict(self.act_net.state_dict())

    def update_act(self):
        if len(self.cache) < self.batch_size:
            return

        minibatch = random.sample(self.cache, self.batch_size)
        # TODO: the memory used maybe optimized better, it should to redesign the data stream and data store
        state = torch.zeros((self.batch_size,self.n_observe)).to(self.device)
        action = torch.zeros((self.batch_size,1),dtype=torch.long).to(self.device)
        action_prob = torch.zeros((self.batch_size,self.n_action)).to(self.device)
        reward = torch.zeros((self.batch_size,1)).to(self.device)
        next_state = torch.zeros((self.batch_size,self.n_observe)).to(self.device)
        for i in range(len(minibatch)):
            state_, action_, action_prob_, reward_, next_state_ = minibatch[i]
            state[i] = state_
            action[i] = action_
            action_prob[i] = to_tensor(action_prob_,self.device)
            reward[i] = reward_
            next_state[i] = next_state_
        prob, value = self.act_net(state)
        # prob_, value_ = self.tgt_net(next_state)

        # pred_values = prob.gather(1, action).squeeze(-1)
        # tgt_values = prob_.max(1)[0].detach() * self.gamma + reward.squeeze(-1)
        # loss = self.criterion(pred_values, tgt_values)

        # # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2 (Note: the L2 penalty is incorporated in optimizer)
        value_loss = self.criterion(value, reward)
        policy_loss = -torch.mean(torch.sum(action_prob * torch.log(prob), 1))
        loss = value_loss + policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        # clip_grad_norm_(self.act_net.parameters(), self.clip_grad)
        self.optimizer.step()
        return loss.item()

    def act_predict(self, state):
        with torch.no_grad():
            action_prob, value = self.act_net(state.unsqueeze(0))
            return action_prob, value

    # def tgt_predict(self, state):
    #     with torch.no_grad():
    #         action_prob, value = self.tgt_net(state.unsqueeze(0))
    #         return action_prob, value

    def save(self, path, i):
        model_path = path + "model_" + str(i) + ".pt"
        torch.save({
            'model_state': self.act_net.state_dict(),
        }, model_path)

    def load(self, path):
        model = torch.load(path)
        self.act_net.load_state_dict(model['model_state'])
        # self.tgt_net.load_state_dict(model['model_state'])
