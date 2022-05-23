from wrappers import to_tensor, obs2tensor
import torch
from bmgame import BMGame

class DQNBMAgent():
    def __init__(self, model, device):
        self.model = model
        self.device =device
        self.game = BMGame()
    
    def predict(self, obs, current_player):

        combine_state = obs2tensor(obs,current_player).to(self.device)
        available_actions = self.game.check(obs, current_player)

        action_prob, value = self.model.predict(combine_state)
        action_prob_mask = to_tensor([action_prob[0][i] for i in available_actions], 'cpu') # store in cpu device

        action_prob = action_prob_mask/torch.sum(action_prob_mask)
        return action_prob, value.item(), combine_state, available_actions



