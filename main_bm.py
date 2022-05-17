import torch
import sys
import argparse

from trainer_bm import Trainer
from wrappers import make_env

from absl import flags

from dqn_agent import DQNBMAgent
from dqn_model import DQNAgent

FLAGS = flags.FLAGS
FLAGS(sys.argv[:1])

parser = argparse.ArgumentParser(description='DQN_MCTS for SC2 BuildMarines',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', choices=['train', 'test'], default='train', help='running mode')
# model parameters
parser.add_argument('--hidden-size', type=int, default=256, help='hidden size')
parser.add_argument('--memory-size', type=int, default=10000, help='size of replay memory')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--eps-start', type=float, default=0.9, help='eps start')
parser.add_argument('--eps-decay', type=float, default=200, help='eps decay step')
parser.add_argument('--eps-end', type=float, default=0.05, help='eps end')
parser.add_argument('--clip-grad', type=float, default=1.0, help='clipping threshold')
# training parameters
parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
parser.add_argument('--numIters', type=int, default=500, help='Total number of training iterations')
parser.add_argument('--num_simulations', type=int, default=5000, help='Total number of MCTS simulations to run when deciding on a move to play')
parser.add_argument('--numEps', type=int, default=200, help='Number of full games (episodes) to run during each iteration')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--log-step', type=int, default=100, help='logging print step')
parser.add_argument('--update-tgt', type=int, default=1, help='update target net')
parser.add_argument('--render', type=int, default=0, help='whether render')
# saving & checkpoint
parser.add_argument('--save-path', type=str, default='model.pt', help='model path for saving')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint for resuming training')
parser.add_argument('--model-path', type=str, default='model.pt', help='model path for evaluation')
parser.add_argument('--test-epoch', type=int, default=3, help='number of test epochs')
args = parser.parse_args()


# args = {
#     'batch_size': 64,
#     'numIters': 500,                                # Total number of training iterations
#     'num_simulations': 1,                         # Total number of MCTS simulations to run when deciding on a move to play
#     'numEps': 16,                                  # Number of full games (episodes) to run during each iteration
#     'numItersForTrainExamplesHistory': 20,
#     'epochs': 2,                                    # Number of epochs of training per iteration
#     'checkpoint_path': 'latest.pth'                 # location to save latest set of weights
# }

# game = Connect2Game()
# board_size = game.get_board_size()
# action_size = game.get_action_size()

# model = Connect2Model(board_size, action_size, device)

# trainer = Trainer(game, model, args)
# trainer.learn()



if __name__ == '__main__':
    # 生成4个env,copy用于simulation
    env = make_env(args)
    env_mirror = make_env(args)
    env_copy = make_env(args)
    env_mirror_copy = make_env(args)
    env_lsit = [[env, env_mirror], [env_copy, env_mirror_copy]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on {device.type}')

    n_state = 18  # 2*BMState
    n_action = 6  # BMAction
    model = DQNAgent(n_state, args.hidden_size, n_action, device, lr=args.lr, batch_size=args.batch_size,
                    memory_size=args.memory_size, gamma=args.gamma, clip_grad=args.clip_grad)
    # 生成2个agent,copy用于simulation
    agent = DQNBMAgent(model, device)
    agent_copy = DQNBMAgent(model, device)
    agent_list = [agent, agent_copy]

    observation_spec = env.observation_spec()[0]
    action_spec = env.action_spec()[0]
    agent.setup(observation_spec, action_spec)

    trainer = Trainer(env_lsit, agent_list, args, device)
    if args.mode == 'train':
        trainer.learn()
    else:
        trainer.evaluate(args.test_epoch)
    env.close()
    env_mirror.close()
