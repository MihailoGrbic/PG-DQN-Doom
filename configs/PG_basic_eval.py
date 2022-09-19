from util import dotdict

args = dotdict({
    'config_path': "scenarios/basic.cfg",
    'scenario_path' : "scenarios/basic.wad",

    'mode' : "eval",                            # "train" for training, "eval" for evaluation.
    'alg' : "PG",                               # Which RL algorithm to use for training and eval. 
                                                # "PG" for Policy Gradient, "DQN" for Deep Q Learning.

    'frame_dim' : (80, 80),                     # Frames are resized to this value before being processed by the net.
    'stack_size' : 4,                           # How many previous frames are stacked and inputed to the net.

    'iterations' : 10,
    'episodes_per_iteration' : 5,

    'load_checkpoint' : True,                   # Must be true if mode is eval.
    'load_path' : "checkpoints/PG_basic1/checkpoint_1000.pth",
})