from bmgame import BMGame
from dqn_agent import DQNBMAgent
from dqn_model import DQNAgent
import os 
import random

env = BMGame()
model_path = './checkpoint/'
model_list = os.listdir(model_path)
my_model = DQNAgent(18, 256, 5, 'cpu')
my_model.load(os.path.join(model_path, model_list[-1]))
my_agent = DQNBMAgent(my_model, 'cpu')
op_model = DQNAgent(18, 256, 5, 'cpu')
op_model.load(os.path.join(model_path,random.choice(model_list[:-1])))
op_agent = DQNBMAgent(op_model, 'cpu')

def evaluate(agent1, agent2):
    obs = env.reset()
    player = 1
    gameover = False
    while not gameover:
        if player == 1:
            action_prob, _, _, available_actions = agent1.predict(obs, player)
        elif player == -1:
            action_prob, _, _, available_actions = agent2.predict(obs, player)
        max_prob = action_prob.argmax(-1)
        action = available_actions[max_prob]
        obs, _ = env.step(obs, action, player)
        gameover = env.check_gameover(obs)
        player *= -1
    value = env.get_value(obs, player)
    print(obs)
    if value > 0:
        print("winner is ",player)
    elif value == 0:
        print("draw")
    elif value < 0:
        print("winner is ",player*-1)

evaluate(my_agent, op_agent)