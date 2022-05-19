import torch
import math
import numpy as np
from marine_agent import BMAction
from copy import deepcopy, copy
from wrappers import env_reset

def ucb_score(parent, child):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score


class Node:
    def __init__(self, prior, current_player, obss=None):
        self.visit_count = 0
        self.current_player = current_player
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None
        self.obs_s = obss              # [obs, obs_mirror]
        self.action_list = []

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        # 对每一个能到达的子节点，用 UCB 进行评估，返回当前具有最大UCB值的子节点及对应动作
        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, state, current_player, action_probs):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        self.current_player = current_player
        self.state = state
        for a, prob in enumerate(action_probs):
            if prob != 0:
                # 生成所有可能的子节点，并赋予对应的先验
                # 对手,需要反转player
                self.children[a] = Node(prior=prob, current_player=self.current_player * -1)

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())


class MCTS:

    def __init__(self, env, env_mirror, agent, args):
        self.env = env
        self.env_mirror = env_mirror
        self.agent = agent
        self.args = args

    def run(self, obs, obs_mirror, current_player):

        env = self.env
        env_mirror = self.env_mirror
        obss = [obs, obs_mirror]
        root = Node(0, current_player, obss=obss)
        # EXPAND root
        # 网络输出先验, 初始 current_player=1
        # action_probs 已合法归一化
        action_probs, _ , combine_state = self.agent.predict(root.obs_s, root.current_player)
        # 传递先验概率给节点，供 MCTS 参考
        # envs , obs_s , agent未更新，传递给 children ; combine_state 传递给 root
        root.expand(combine_state, current_player, action_probs)

        loop = False
        # 在最大模拟推演数限制下进行模拟（不一定模拟到终局）
        for _ in range(self.args.num_simulations):
            if loop:
                # env and agent initialize
                obs, obs_mirror, env, env_mirror, agent = env_reset(self.env, self.env_mirror, self.agent, self.args)

            node = root
            search_path = [node]
            # SELECT
            # 只要还有未记录的可扩展子节点，持续执行
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)
            # 选中子节点和对应动作后，同步env到该state，做好执行该动作的准备
            for i, act in enumerate(node.action_list):
                # 在执行动作时，agent内记录对局情况的参数也会相应改变
                # 我方
                if i%2==0:
                    obs = env.step(actions=[act])[0]
                # 对手
                else:
                    obs_mirror = env_mirror.step(actions=[act])[0]
            # 定义当前父节点, 执行最优子节点动作
            parent = search_path[-2]
            child = search_path[-1]
        
            # 替当前父节点执行动作，符号 action 转为真实环境 action_real
            # 将获得的新状态记录下来，给予对应子节点
            if parent.current_player == 1:
                mirror = False
                #TODO: 不一定要把待执行的动作，提交到in_progress中
                self.agent.in_progress = BMAction(action)
                if self.agent.in_progress == BMAction.NO_OP:
                    action_real, _ = self.agent.choose_act(obs, mirror)
                    obs = env.step(actions=[action_real])[0]
                    child.action_list.append(action_real)
                while self.agent.in_progress != BMAction.NO_OP:
                    action_real, _ = self.agent.choose_act(obs, mirror)
                    obs = env.step(actions=[action_real])[0]
                    child.action_list.append(action_real)
            elif parent.current_player == -1:
                mirror = True
                self.agent.in_progress_mirror = BMAction(action)
                if self.agent.in_progress_mirror == BMAction.NO_OP:
                    action_real_mirror, _ = self.agent.choose_act(obs_mirror, mirror)
                    obs_mirror = env_mirror.step(actions=[action_real_mirror])[0]
                    child.action_list.append(action_real_mirror)
                while self.agent.in_progress_mirror != BMAction.NO_OP:
                    action_real_mirror, _ = self.agent.choose_act(obs_mirror, mirror)
                    obs_mirror = env_mirror.step(actions=[action_real_mirror])[0]
                    child.action_list.append(action_real_mirror)
            
            # 执行动作后，更新当前的子节点信息
            child.obs_s = [obs, obs_mirror]
            # # 在expand时会将next_state赋予目前的子节点
            # next_state = child.agent.get_combine_state(child.obs_s, child.current_player)
            # 判断对局是否到达终点，计算最终胜负得分
            value = self.agent.cal_win_value(child.obs_s)
            if value is None:
                # 从对手的视角看，当前状态转移到达状态的价值（比如这一步你走得好，那么对手眼中，这个状态价值就低）
                action_probs, value , combine_state= agent.predict(child.obs_s, child.current_player)
                child.expand(combine_state, child.current_player, action_probs)
            self.backpropagate(search_path, value, parent.current_player * -1)

        return root

    def backpropagate(self, search_path, value, current_player):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.current_player == current_player else -value
            node.visit_count += 1
