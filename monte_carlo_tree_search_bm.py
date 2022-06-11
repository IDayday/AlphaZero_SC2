import torch
import math
import numpy as np
from copy import deepcopy
from wrappers import obs2tensor

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
    def __init__(self, prior, current_player, obs=None):
        self.visit_count = 0
        self.current_player = current_player
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.obs = obs
        self.state = None

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

        # evaluate each children node by UCB , return the best
        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, current_player, action_probs, available_actions):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        # a is the index, but available_actions[a] is the true action
        for a, prob in enumerate(action_probs):
            if prob != 0:
                self.children[available_actions[a]] = Node(prior=prob, current_player=current_player * -1)

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())


class MCTS:

    def __init__(self, env, obs, agent, args, device):
        self.env = env
        self.obs = obs
        self.agent = agent
        self.args = args
        self.device = device

    def run(self, current_player, max_search):

        root = Node(0, current_player, obs=self.obs) # set the beginning root by obs
        # EXPAND root
        # predict the normalized action_probs
        root_action_probs, _ , combine_state, root_available_actions = self.agent.predict(root.obs, root.current_player)
        root.state = combine_state
        root.expand(current_player, root_action_probs, root_available_actions)

        # simulate under the maxsteps
        for _ in range(max_search):
            node = root
            search_path = [node]
            # SELECT
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)
            # setting the parent node and child node
            parent = search_path[-2]
            child = search_path[-1]
        
            # get the new obs/state and put it in the child node
            obs, _ = self.env.step(parent.obs, action, parent.current_player)
            child.obs = deepcopy(obs)
            # check if the game is over
            gameover = self.env.check_gameover(obs)
            # the value is from the parent's perspective, but it store in the child node
            # so, if we check the value in a node, remember that it reflects your opponent's view
            value = self.env.get_value(obs, parent.current_player) if gameover else None
            if value == None:
                action_probs, value, combine_state, available_actions = self.agent.predict(child.obs, child.current_player)
                child.state = combine_state
                child.expand(child.current_player, action_probs, available_actions)
            self.backpropagate(search_path, value, parent.current_player * -1)

        return root, root_action_probs, root_available_actions

    def backpropagate(self, search_path, value, current_player):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.current_player == current_player else -value
            node.visit_count += 1
