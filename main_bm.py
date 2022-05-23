import torch
import sys
import argparse

from trainer_bm import Trainer

from bmgame import BMGame
from dqn_agent import DQNBMAgent
from dqn_model import DQNAgent
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

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
parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')
parser.add_argument('--numIters', type=int, default=500, help='Total number of training iterations')
parser.add_argument('--num_simulations', type=int, default=200, help='Total number of MCTS simulations to run when deciding on a move to play')
parser.add_argument('--numEps', type=int, default=200, help='Number of full games (episodes) to run during each iteration')
parser.add_argument('--batch-size', type=int, default=256, help='batch size')
parser.add_argument('--log-step', type=int, default=100, help='logging print step')
parser.add_argument('--update_tgt', type=int, default=20, help='update target net')
parser.add_argument('--render', type=int, default=0, help='whether render')
# saving & checkpoint
parser.add_argument('--save-path', type=str, default='model.pt', help='model path for saving')
parser.add_argument('--checkpoint_iter', type=int, default=1, help='checkpoint iterations')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint for resuming training')
parser.add_argument('--model_path', type=str, default='./checkpoint/', help='model path for evaluation')
parser.add_argument('--test-epoch', type=int, default=3, help='number of test epochs')
args = parser.parse_args()


if __name__ == '__main__':
    env = BMGame()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on {device.type}')

    n_state = 18
    n_action = 5
    # algorithm DQN
    model = DQNAgent(n_state, args.hidden_size, n_action, device, lr=args.lr, batch_size=args.batch_size,
                    memory_size=args.memory_size, gamma=args.gamma, clip_grad=args.clip_grad)
    agent = DQNBMAgent(model, device)

    trainer = Trainer(env, agent, args, device)
    if args.mode == 'train':
        trainer.learn()
    else:
        trainer.evaluate(args.test_epoch)
