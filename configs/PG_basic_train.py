from util import dotdict

args = dotdict({
    'config_path': "scenarios/basic.cfg",
    'scenario_path' : "scenarios/basic.wad",

    'mode' : "train",                           # "train" for training, "eval" for evaluation.
    'alg' : "PG",                               # Which RL algorithm to use for training and eval. 
                                                # "PG" for Policy Gradient, "DQN" for Deep Q Learning.

    'frame_dim' : (80, 80),                     # Frames are resized to this value before being processed by the net.
    'stack_size' : 4,                           # How many previous frames are stacked and inputed to the net.
    'gamma' : 1,                                # Reward decay rate.
    'normalize_rewards' : False,                # Whether or not to normalize rewards across episodes for PG.

    'iterations' : 2000,
    'episodes_per_iteration' : 4,
    'epochs' : 1,
    'batch_size' : 512,
    'batches_per_epoch' : 3,                    # How many batches are sampled from the replay buffer for training per epoch.
                                                # Set to 0 to train on the whole replay buffer.
    'fill_history_at_start' : True,            # If true will run episodes at the start until the replay buffer is full.
    'saved_episodes' : 80,                      # How many episodes are saved in a replay buffer and used for training.
                                                # Episodes are saved instead of individual states for reward
                                                # normalization in PG. Large values can overflow RAM memory.
    'lr' : 1e-6,
    'lr_decay' : 0.1,                           # How much does the learning rate decay at each milestone. Set to 1 to disable.
    'lr_milestones' : [1000],                   # Iterations at which the learning rate decays.
    'dropout' : 0.3,

    'save_every_x_iters' : 20,
    'checkpoint_folder' : "checkpoints/PG_basic/",
    'checkpoint_filename' : "checkpoint",       # Filename without any extensions.

    'load_checkpoint' : False,                  # Must be true if mode is eval.
    'load_path' : "checkpoints/sandboxPG/checkpoint_1000.pth",
    'resume_iter' : 0                           # Iteration from which the training is resumed.
})