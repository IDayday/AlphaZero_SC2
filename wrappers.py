from pysc2.env import sc2_env
import torch


def make_env(args):
    return sc2_env.SC2Env(
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(
                screen=84,
                minimap=32,
            )
        ),
        map_name='BuildMarines',
        step_mul=8,
        visualize=args.render,
    )

def to_tensor(data, device):
    return torch.tensor(data, dtype=torch.float, device=device)

def check_bug_initiate(obs, env, agent, args, mirror):
    if obs.observation.player.idle_worker_count > 0:
        # bug happens!
        print('A bug happened! Mineral missing! Restart enviroment')
        env.close()
        env = make_env(args)
        obs = env.reset()[0]
    # initiate
    if not mirror:
        while agent.in_progress == -1:
            action = agent.initiating(obs, mirror)
            obs = env.step(actions=[action])[0]
        return obs
    elif mirror:
        while agent.in_progress_mirror == -1:
            action = agent.initiating(obs, mirror)
            obs = env.step(actions=[action])[0]
        return obs             

# 游戏环境初始化 
def env_reset(env, env_mirror, agent, args):
    obs = env.reset()[0]
    obs_mirror = env_mirror.reset()[0]
    # agent也要初始化一次
    agent.reset() 
    # DEBUG initiate之后SCV没有开始采矿
    obs = check_bug_initiate(obs, env, agent, args, mirror=False)
    obs_mirror = check_bug_initiate(obs_mirror, env_mirror, agent, args, mirror=True)
    return obs, obs_mirror, env, env_mirror, agent
