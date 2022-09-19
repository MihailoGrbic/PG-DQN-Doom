import os
import copy
import csv

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from util import process_frame, stack_frames, disc_rewards, norm_rewards

def sample_trajectories(net, game, possible_actions, args):
    net.eval() # Sets the network to inference mode
    sample_results = []
    
    # Start a new episode
    game.new_episode()
    curr_episode = 0

    episode_states, episode_actions, episode_rewards = [], [], []
    frame_stack = []
    while curr_episode < args.episodes_per_iteration:
        # Get a new observation
        frame = process_frame(game.get_state().screen_buffer, args.frame_dim)
        
        frame_stack, np_frame_stack = stack_frames(frame, args.stack_size, frame_stack)

        # Run state through the NN
        state = torch.tensor(np_frame_stack).float().to('cuda')
        net_result = net(state.reshape((1, args.stack_size, *args.frame_dim)))

        if args.alg == "PG":
            action_prob = net_result.cpu().detach().numpy()
            # Select action w.r.t the actions prob
            action = np.random.choice(range(action_prob.shape[1]), p=action_prob.ravel()) 
        elif args.alg == "DQN":
            Q_values = net_result.cpu().detach().numpy()
            # Select action with epsilon greedy strategy
            if np.random.rand() < args.epsilon: action = np.random.choice(range(Q_values.shape[1]))
            else: action = np.argmax(Q_values)
            if "epsilon_decay" in args: args.epsilon = max(args.epsilon * args.epsilon_decay, args.epsilon_final)
            
        action = possible_actions[action]
        # Perform action
        reward = game.make_action(action)
        done = game.is_episode_finished()

        # Store results
        episode_states.append(np_frame_stack)
        episode_actions.append(action)
        episode_rewards.append(reward)
        
        # Check if the episode is finished
        if done:
            sample_results.append((np.array(episode_states), np.array(episode_actions), 
                            np.array(episode_rewards)))     
            
            episode_states, episode_actions, episode_rewards = [], [], []
            frame_stack = []
            # Start a new episode
            game.new_episode()
            curr_episode += 1

    return sample_results

def trainPG(net, game, possible_actions, args):
    # Set current iter if resuming from a saved model
    iter = args.resume_iter if args.load_checkpoint else 0

    optimizer = optim.Adam(net.parameters(), args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_milestones, gamma=args.lr_decay, last_epoch=iter-1)
    scheduler.last_epoch = iter-1
    
    episode_replay_buffer = []
    # Fill the replay buffer at the start if needed
    if args.fill_history_at_start:
        while len(episode_replay_buffer) < args.saved_episodes - args.episodes_per_iteration:
            episode_replay_buffer += sample_trajectories(net, game, possible_actions, args)

    while iter < args.iterations:
        iter += 1

        # Sample trajectories and save the results
        sample_results = sample_trajectories(net, game, possible_actions, args)
        episode_replay_buffer = episode_replay_buffer + sample_results
        episode_replay_buffer = episode_replay_buffer[-args.saved_episodes:]

        # Analytics
        # Calculate the total reward of each episode from this iteration
        total_ep_rewards = [np.sum(episode[2]) for episode in sample_results]

        print("==========================================")
        print("Iteration: ", iter, "/", args.iterations)
        print("-----------")
        print("Episodes per iteration: {}".format(args.episodes_per_iteration))
        print("Episodes in the replay buffer: {}".format(len(episode_replay_buffer)))
        print("Min iteration reward: {}".format(min(total_ep_rewards)))
        print("Average iteration reward: {}".format(np.mean(total_ep_rewards)))
        print("Max iteration reward: {}".format(max(total_ep_rewards)))

        with open('results.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([min(total_ep_rewards), np.mean(total_ep_rewards), max(total_ep_rewards)])

        # Prepare all data for training
        states, actions, discounted_rewards = [], [], []
        for episode in episode_replay_buffer:
            states.append(episode[0])
            actions.append(episode[1])
            discounted_rewards.append(disc_rewards(episode[2], args.gamma))

        # Normalize all discounted rewards
        if args.normalize_rewards: discounted_rewards = norm_rewards(discounted_rewards)

        # Flatten everything into vectors for training
        states, actions, discounted_rewards = np.concatenate(states), np.concatenate(actions), np.concatenate(discounted_rewards)

        # Create a dataset from the states, actions, reward triplets
        dataset = list(zip(states, actions, discounted_rewards))

        # Training
        net.train() # Set the network to training mode
        for epoch in range(args.epochs):
            print("EPOCH: ", epoch + 1, "/", args.epochs)
            epoch_loss = 0
            
            # Calculate the number of training examples used for this epoch
            epoch_length = len(dataset) if args.batches_per_epoch == 0 \
                        else min(len(dataset), args.batches_per_epoch * args.batch_size)
            # Sample epoch_length training examples from all available training examples
            indexes = np.random.choice(len(dataset), size=epoch_length, replace=False)
            dataset_sample = [dataset[i] for i in indexes]

            # Create a dataloader to divide training examples into batches
            dataloader = DataLoader(dataset_sample, batch_size=args.batch_size, shuffle=True)

            for states, actions, discounted_rewards in tqdm(dataloader):
                states = states.float().cuda()
                actions = actions.cuda()
                discounted_rewards = discounted_rewards.cuda()

                probs = net(states)
                log_probs = -torch.log(probs[actions==1])   # "-" because PG uses gradient ascent
                # Divide the loss by the number of episodes used for training, since we can't devide dJ0 directly
                loss = torch.sum(log_probs * discounted_rewards) / len(episode_replay_buffer)

                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("Epoch Loss: {}".format(epoch_loss/len(dataloader)))

        scheduler.step() # LR scheduler step at the end of an iteration

        # Save the model
        if iter % args.save_every_x_iters==0:
            if not os.path.exists(args.checkpoint_folder):
                print("Checkpoint Directory does not exist! Making directory {}".format(args.checkpoint_folder))
                os.mkdir(args.checkpoint_folder)

            path = os.path.join(args.checkpoint_folder, args.checkpoint_filename+"_"+str(iter).rjust(2, '0')+".pth")
            torch.save(net.state_dict(), path)
    
def trainDQN(net, game, possible_actions, args):
    # Set current iter if resuming from a saved model
    iter = args.resume_iter if args.load_checkpoint else 0

    optimizer = optim.Adam(net.parameters(), args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_milestones, gamma=args.lr_decay)
    scheduler.last_epoch = iter-1

    # Set epsilon
    args.epsilon = args.epsilon_init * args.epsilon_decay ** (iter * 100)

    episode_replay_buffer = []
    # Fill the replay buffer at the start if needed
    if args.fill_history_at_start:
        # Disable epsilon decay while filling the replay buffer at the start
        epsilon_decay = args.epsilon_decay
        args.epsilon_decay = 1
        while len(episode_replay_buffer) < args.saved_episodes - args.episodes_per_iteration:
            episode_replay_buffer += sample_trajectories(net, game, possible_actions, args)
        args.epsilon_decay = epsilon_decay

    tau = 0
    target_net = copy.deepcopy(net)
    target_net.to('cuda')
    target_net.eval()

    while iter < args.iterations:
        iter += 1

        # Sample trajectories and save the results
        sample_results = sample_trajectories(net, game, possible_actions, args)
        episode_replay_buffer = episode_replay_buffer + sample_results
        episode_replay_buffer = episode_replay_buffer[-args.saved_episodes:]

        # Analytics
        # Calculate the total reward of each episode from this iteration
        total_ep_rewards = [np.sum(episode[2]) for episode in sample_results]

        print("==========================================")
        print("Iteration: ", iter, "/", args.iterations)
        print("-----------")
        print("Episodes per iteration: {}".format(args.episodes_per_iteration))
        print("Episodes in the replay buffer: {}".format(len(episode_replay_buffer)))
        print("Current epsilon value: {}".format(args.epsilon))
        print("Min iteration reward: {}".format(min(total_ep_rewards)))
        print("Average iteration reward: {}".format(np.mean(total_ep_rewards)))
        print("Max iteration reward: {}".format(max(total_ep_rewards)))

        # Prepare all data for training
        states, next_states, actions, rewards, ends = [], [], [], [], []
        for episode in episode_replay_buffer:
            states.append(episode[0])
            actions.append(episode[1])
            rewards.append(episode[2])
            # Create a next states vector by shifting the state vector to the left
            next_states.append(np.append(episode[0][1:], episode[0][0:1], axis=0))
            # Create a bool vector for episode ends
            end_vec = np.zeros_like(episode[2])
            end_vec[-1] = 1
            ends.append(end_vec)

        # Flatten everything into vectors for training
        states, next_states, actions,  = np.concatenate(states), np.concatenate(next_states), np.concatenate(actions)
        rewards, ends = np.concatenate(rewards), np.concatenate(ends)

        # Create a dataset
        dataset = list(zip(states, next_states, actions, rewards, ends))

        # Training
        net.train() # Set the network to training mode
        for epoch in range(args.epochs):
            print("EPOCH: ", epoch + 1, "/", args.epochs)
            epoch_loss = 0
            
            # Calculate the number of training examples used for this epoch
            epoch_length = len(dataset) if args.batches_per_epoch == 0 \
                        else min(len(dataset), args.batches_per_epoch * args.batch_size)
            # Sample epoch_length training examples from all available training examples
            indexes = np.random.choice(len(dataset), size=epoch_length, replace=False)
            dataset_sample = [dataset[i] for i in indexes]

            # Create a dataloader to divide training examples into batches
            dataloader = DataLoader(dataset_sample, batch_size=args.batch_size, shuffle=True)

            for states, next_states, actions, rewards, ends in tqdm(dataloader):
                states = states.float().cuda()
                next_states = next_states.float().cuda()
                actions = actions.cuda()
                rewards = rewards.cuda()
                ends = ends.cuda()

                curr_Q_values = net(states)
                next_Q_values = target_net(next_states)
                Q_target = rewards
                Q_target += torch.mul(args.gamma, torch.max(next_Q_values, dim=1)[0]) * (1 - ends)
                loss = torch.sum(torch.square(Q_target - curr_Q_values[actions==1]))

                epoch_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update target net
                tau += 1
                if tau == args.q_tg_update_step:
                    tau = 0
                    target_net.load_state_dict(net.state_dict())

            print("Epoch Loss: {}".format(epoch_loss/len(dataloader)))

        scheduler.step() # LR scheduler step at the end of an iteration

        # Save the model
        if iter % args.save_every_x_iters==0:
            if not os.path.exists(args.checkpoint_folder):
                print("Checkpoint Directory does not exist! Making directory {}".format(args.checkpoint_folder))
                os.mkdir(args.checkpoint_folder)

            path = os.path.join(args.checkpoint_folder, args.checkpoint_filename+"_"+str(iter).rjust(2, '0')+".pth")
            torch.save(net.state_dict(), path)

def eval(net, game, possible_actions, args):
    iter = 0
    while True:
        iter += 1
        args.epsilon = 0

        # Sample trajectories and save the results
        sample_results = sample_trajectories(net, game, possible_actions, args)

        # Analytics
        # Calculate the total reward of each episode from this iteration
        total_ep_rewards = [np.sum(episode[2]) for episode in sample_results]

        print("==========================================")
        print("Eval iteration: ", iter)
        print("-----------")
        print("Episodes per iteration: {}".format(args.episodes_per_iteration))
        print("Min iteration reward: {}".format(min(total_ep_rewards)))
        print("Average iteration reward: {}".format(np.mean(total_ep_rewards)))
        print("Max iteration reward: {}".format(max(total_ep_rewards)))

def load_model(net, model_path, verbose=False):
    if verbose:print("Loading checkpoint: " + model_path)
    net.load_state_dict(torch.load(model_path))