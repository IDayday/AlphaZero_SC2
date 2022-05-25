from wrappers import to_tensor, obs2tensor
import torch
from bmgame import BMGame
import random
import numpy as np

class DQNBMAgent():
    def __init__(self, model, device, args=None):
        self.model = model
        self.device = device
        self.args = args
        self.game = BMGame()
    
    def predict(self, obs, current_player):

        combine_state = obs2tensor(obs,current_player).to(self.device)
        available_actions = self.game.check(obs, current_player)

        # act network VS target network
        # if current_player == 1:
        #     action_prob, value = self.model.act_predict(combine_state)
        # elif current_player == -1:
        #     action_prob, value = self.model.tgt_predict(combine_state)
        
        # act network VS random
        # if current_player == 1:
        #     action_prob, value = self.model.act_predict(combine_state)
        # elif current_player == -1:
        #     action_prob, value = torch.randn((1,5)), torch.tensor([0])

        # act network VS act network(noise)
        action_prob, value = self.model.act_predict(combine_state)
        if self.args.noise and current_player == -1:
            epsilon = 0.2
            # different from paper, in the paper, noise is added to the root of MCTS Tree
            # Here, noise is just added to the result
            noise_distri = np.random.dirichlet(0.3 * np.ones(len(action_prob)))
            noise_distri = torch.from_numpy(noise_distri).float().to(self.device)
            action_prob = (1 - epsilon) * action_prob + epsilon * noise_distri

        action_prob_mask = to_tensor([action_prob[0][i] for i in available_actions], 'cpu') # store in cpu device

        action_prob = action_prob_mask/torch.sum(action_prob_mask)
        return action_prob, value.item(), combine_state, available_actions



class DQNBMAgent_E():
    def __init__(self, model, device, args=None):
        self.model = model
        self.device =device
        self.args = args
        self.game = BMGame()
    
    def predict(self, obs, current_player):

        combine_state = obs2tensor(obs,current_player).to(self.device)
        available_actions = self.game.check(obs, current_player)

        action_prob, value = self.model.act_predict(combine_state)
        action_prob_mask = to_tensor([action_prob[0][i] for i in available_actions], 'cpu') # store in cpu device

        action_prob = action_prob_mask/torch.sum(action_prob_mask)
        return action_prob, value.item(), combine_state, available_actions