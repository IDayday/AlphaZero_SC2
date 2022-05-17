from marine_agent import BaseBMAgent, BMAction
from wrappers import to_tensor
import torch


class DQNBMAgent(BaseBMAgent):
    def __init__(self, model, device):
        BaseBMAgent.__init__(self)
        self.model = model
        self.device =device
    
    def cal_win_value(self,obs_s):
        obs, obs_mirror = obs_s
        minerals = obs.observation['score_cumulative'][7]
        minerals_mirror = obs_mirror.observation['score_cumulative'][7]
        total_minerals = minerals + minerals_mirror
        if obs.last() or obs_mirror.last() or total_minerals >= 20000:
            if self.reward > self.reward_mirror:
                return 500
            elif self.reward < self.reward_mirror:
                return -500
            else:
                return 0
        else:
            return None

    
    def get_combine_state(self, obs_s, current_player):
        obs, obs_mirror = obs_s
        if current_player == 1:
            combine_state = torch.cat((to_tensor(self.get_state(obs),self.device), \
                            -to_tensor(self.get_state(obs_mirror),self.device))) 
        else:
            combine_state = torch.cat((to_tensor(self.get_state(obs_mirror),self.device), \
                            -to_tensor(self.get_state(obs),self.device)))
        return combine_state
    
    def predict(self, obs_s, current_player):
        obs, obs_mirror = obs_s
        if current_player == 1:
            mirror = False
            player = obs.observation['player']
            combine_state = torch.cat((to_tensor(self.get_state(obs),self.device), \
                            -to_tensor(self.get_state(obs_mirror),self.device))) 
            checkers = [
                        (lambda x,y: True),
                        self.check_make_scv,
                        self.check_build_depot,
                        self.check_build_barracks,
                        self.check_make_marine,
                        self.check_kill_marine
                    ]
            # 有效动作集
            choices = [i for i in range(self.n_action) if checkers[i](player, mirror)]
            if self.in_progress == BMAction.NO_OP:
                action_prob, value= self.model.predict(combine_state)
                action_prob_mask = to_tensor([action_prob[0][i] for i in choices], self.device)
                action_prob = action_prob_mask/torch.sum(action_prob_mask)
        
        else:
            mirror = True
            player = obs_mirror.observation['player']
            combine_state = torch.cat((to_tensor(self.get_state(obs_mirror),self.device), \
                            -to_tensor(self.get_state(obs),self.device)))
            checkers = [
                        (lambda x,y: True),
                        self.check_make_scv,
                        self.check_build_depot,
                        self.check_build_barracks,
                        self.check_make_marine,
                        self.check_kill_marine
                    ]
            # 有效动作集
            choices = [i for i in range(self.n_action) if checkers[i](player, mirror)]
            if self.in_progress_mirror == BMAction.NO_OP:
                action_prob, value= self.model.predict(combine_state)
                action_prob_mask = to_tensor([action_prob[0][i] for i in choices], self.device)
                action_prob = action_prob_mask/torch.sum(action_prob_mask)

        return action_prob, value, combine_state

    def step(self, obs, obs_mirror, current_player):
        # 我方回合是 current_player : 1
        if current_player == 1:
            mirror = False
            super().step(obs, mirror)
            player = obs.observation['player']
            combine_state = torch.cat((to_tensor(self.get_state(obs),self.device), \
                            -to_tensor(self.get_state(obs_mirror),self.device))) 
            checkers = [
                        (lambda x: True),
                        self.check_make_scv,
                        self.check_build_depot,
                        self.check_build_barracks,
                        self.check_make_marine,
                        self.check_kill_marine
                    ]
            # 有效动作集
            choices = [i for i in range(self.n_action) if checkers[i](player, mirror)]
            if self.in_progress == BMAction.NO_OP:
                action_prob, value= self.model.predict(combine_state)
                action_prob_mask = [action_prob[0][i] for i in choices]
                action_prob = action_prob_mask/torch.sum(action_prob_mask)
                value = [value[0][j] for j in choices]
            self.step_reward += min(0, 0.5 - player.minerals / 1000)
        
        else:
            mirror = True
            super().step(obs_mirror, mirror)
            player = obs_mirror.observation['player']
            combine_state = torch.cat((to_tensor(self.get_state(obs_mirror),self.device), \
                            -to_tensor(self.get_state(obs),self.device)))
            checkers = [
                        (lambda x: True),
                        self.check_make_scv,
                        self.check_build_depot,
                        self.check_build_barracks,
                        self.check_make_marine,
                        self.check_kill_marine
                    ]
            # 有效动作集
            choices = [i for i in range(self.n_action) if checkers[i](player, mirror)]
            if self.in_progress_mirror == BMAction.NO_OP:
                action_prob, value= self.model.predict(combine_state)
                action_prob_mask = [action_prob[0][i] for i in choices]
                action_prob = action_prob_mask/torch.sum(action_prob_mask)
                value = [value[0][j] for j in choices]
            self.step_reward_mirror += min(0, 0.5 - player.minerals / 1000)

        return action_prob, value, combine_state

    def initiating(self, obs, mirror):
        act, act_index = self.choose_act(obs, mirror)
        return act

