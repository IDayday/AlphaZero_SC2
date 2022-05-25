# from pysc2.env import sc2_env
import torch
import numpy as np

# def make_env(args):
#     return sc2_env.SC2Env(
#         players=[sc2_env.Agent(sc2_env.Race.terran)],
#         agent_interface_format=sc2_env.AgentInterfaceFormat(
#             feature_dimensions=sc2_env.Dimensions(
#                 screen=84,
#                 minimap=32,
#             )
#         ),
#         map_name='BuildMarines',
#         step_mul=8,
#         visualize=args.render,
#     )

def to_tensor(data, device):
    return torch.tensor(data, dtype=torch.float, device=device)

# def check_bug_initiate(obs, env, agent, args, mirror):
#     if obs.observation.player.idle_worker_count > 0:
#         # bug happens!
#         print('A bug happened! Mineral missing! Restart enviroment')
#         env.close()
#         env = make_env(args)
#         obs = env.reset()[0]
#     # initiate
#     if not mirror:
#         while agent.in_progress == -1:
#             action = agent.initiating(obs, mirror)
#             obs = env.step(actions=[action])[0]
#         return obs
#     elif mirror:
#         while agent.in_progress_mirror == -1:
#             action = agent.initiating(obs, mirror)
#             obs = env.step(actions=[action])[0]
#         return obs             

# # 游戏环境初始化 
# def env_reset(env, env_mirror, agent, args):
#     obs = env.reset()[0]
#     obs_mirror = env_mirror.reset()[0]
#     # agent也要初始化一次
#     agent.reset() 
#     # DEBUG initiate之后SCV没有开始采矿
#     obs = check_bug_initiate(obs, env, agent, args, mirror=False)
#     obs_mirror = check_bug_initiate(obs_mirror, env_mirror, agent, args, mirror=True)
#     return obs, obs_mirror, env, env_mirror, agent

# restructure the obs 
def obs2tensor(obs, player):
    features = []
    n_scv = obs[1][2]
    n_scv_op = obs[5][2]
    n_marine = obs[1][3]
    n_marine_op = obs[5][3]
    n_population = obs[1][1]
    n_population_op = obs[5][1]
    n_population_cap = obs[1][0]
    n_population_cap_op = obs[5][0]
    n_depot = obs[2][0]
    n_depot_op = obs[6][0]
    n_barracks = obs[2][1]
    n_barracks_op = obs[6][1]
    n_mainerals = obs[0][0]
    n_mainerals_op = obs[4][0]
    n_depot_producing = len(obs[3][2])
    n_depot_producing_op = len(obs[7][2])
    n_barracks_producing = len(obs[3][3])
    n_barracks_producing_op = len(obs[3][3])

    # normalize
    if player == 1:
        features= [n_scv/10,n_marine/2,n_population/10,n_population_cap/10, \
                    n_depot/5,n_barracks/2,n_mainerals/1000,n_depot_producing/5,n_barracks_producing/2,\
                    n_scv_op/10,n_marine_op/2,n_population_op/10,n_population_cap_op/10,\
                    n_depot_op/5,n_barracks_op/2,n_mainerals_op/1000,n_depot_producing_op/5,n_barracks_producing_op/2]
    elif player == -1:
        features= [n_scv_op/10,n_marine_op/2,n_population_op/10,n_population_cap_op/10,\
                    n_depot_op/5,n_barracks_op/2,n_mainerals_op/1000,n_depot_producing_op/5,n_barracks_producing_op/2,\
                    n_scv/10,n_marine/2,n_population/10,n_population_cap/10, \
                    n_depot/5,n_barracks/2,n_mainerals/1000,n_depot_producing/5,n_barracks_producing/2]
    features = torch.from_numpy(np.array(features)).float()

    return features