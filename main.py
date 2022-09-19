import torch
import numpy as np

from train import trainPG, trainDQN, eval, load_model
from model import PGNet, DQNet
from environment import create_environment

#HYPERPARAMETERS
from configs.PG_basic_train import args

seed = 3141592 # Set seed for reproducibility, since PG is unstable. 
torch.manual_seed(seed)
np.random.seed(seed)

if __name__ == "__main__":
    game, possible_actions = create_environment(config_path=args.config_path, scenario_path=args.scenario_path, seed=seed)
    
    # If any unimportant values are missing from the config fill them with default values
    args.fill_w_default()

    # Initialize the network
    if args.alg == "PG":
        net = PGNet((args.stack_size, *args.frame_dim), len(possible_actions), args.dropout)
    elif args.alg == "DQN":
        net = DQNet((args.stack_size, *args.frame_dim), len(possible_actions), args.dropout)
    else:
        raise Exception("Unknown training algorithm: " + args.alg)

    if args.load_checkpoint:
        load_model(net, args.load_path, True)
    net.to('cuda')

    if args.mode == "train":
        if args.alg == "PG":
            trainPG(net, game, possible_actions, args) 
        elif args.alg == "DQN":
            trainDQN(net, game, possible_actions, args) 
    elif args.mode == "eval":
        eval(net, game, possible_actions, args)     # Run evaluation
    
    game.close()