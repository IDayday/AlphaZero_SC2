import torch
import sys
import argparse
from trainer_bm2_multi import simulation, learn, evaluate
import random
import time
import torch.multiprocessing as mp 
from wrappers import to_tensor, obs2tensor
from copy import deepcopy
from bmgame import BMGame
from bm_agent import BMAgent, BMAgent_E
from bm_model import Agent
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
torch.multiprocessing.set_sharing_strategy('file_system')
parser = argparse.ArgumentParser(description='DQN_MCTS for SC2 BuildMarines',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', choices=['train', 'test'], default='train', help='running mode')
# model parameters
parser.add_argument('--hidden-size', type=int, default=256, help='hidden size')
parser.add_argument('--memory-size', type=int, default=100000, help='size of replay memory')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--eps-start', type=float, default=0.9, help='eps start')
parser.add_argument('--eps-decay', type=float, default=200, help='eps decay step')
parser.add_argument('--eps-end', type=float, default=0.05, help='eps end')
parser.add_argument('--clip-grad', type=float, default=1.0, help='clipping threshold')
# training parameters
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')
parser.add_argument('--warm_up', type=int, default=50, help='number of warmup epochs')
parser.add_argument('--numIters', type=int, default=500, help='Total number of training iterations')
parser.add_argument('--num_simulations', type=int, default=200, help='Total number of MCTS simulations to run when deciding on a move to play')
parser.add_argument('--batch-size', type=int, default=512, help='batch size')
parser.add_argument('--update_tgt', type=int, default=1, help='update target net')
parser.add_argument('--noise', type=bool, default=True, help='add noise to the action prob when self-play')
parser.add_argument('--num_processes', type=int, default=10, help='number of multiprocesses')
# saving & checkpoint
parser.add_argument('--save-path', type=str, default='model.pt', help='model path for saving')
parser.add_argument('--checkpoint_iter', type=int, default=10, help='checkpoint iterations')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint for resuming training')
parser.add_argument('--model_path', type=str, default='./checkpoint/default_evaluate_max2/', help='model path for evaluation')
parser.add_argument('--test-epoch', type=int, default=3, help='number of test epochs')
args = parser.parse_args()


if __name__ == '__main__':
    env = BMGame()
    if args.num_processes > 1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on {device.type} and {args.num_processes} processes')

    dir_name = args.model_path
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    args.n_state = 18
    args.n_action = 5
    # algorithm
    model = Agent(args.n_state, args.hidden_size, args.n_action, device, lr=args.lr, batch_size=args.batch_size,
                    memory_size=args.memory_size, gamma=args.gamma, clip_grad=args.clip_grad)
    model_param = model.act_net
    model_param.share_memory()

    data_queue = mp.Queue(maxsize=100)   # FIFO 
    signal_queue = mp.Queue()
    simlog_queue = mp.Queue()
    # trainlog_queue = mp.Queue()

    start = time.time()
    processes = [] 
    p = mp.Process(target=learn, args=(model_param, args, device, data_queue, signal_queue, simlog_queue))
    p.start()
    processes.append(p)
    for rank in range(1, args.num_processes + 1):
        p = mp.Process(target=simulation, args=(rank, env, model_param, args, device, data_queue, signal_queue, simlog_queue))
        p.start()
        processes.append(p)

    p = mp.Process(target=evaluate, args=(env, model_param, signal_queue, args))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()