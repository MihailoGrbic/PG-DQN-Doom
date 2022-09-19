import numpy as np
from skimage.transform import resize

def process_frame(frame, frame_dim):
    '''
    Process a frame by removing useless information (the roof), normalizing it, and resizing it
    '''
    cropped_frame = frame[160:,:]               # Remove useless information from the frame
    normalized_frame = cropped_frame/255.0
    preprocessed_frame = resize(normalized_frame, frame_dim)
    return preprocessed_frame

def stack_frames(frame, size, prev_stack=[]):
    '''
    Implement a queue structure of length size for game frames (even though it's called stack_frames)
    If the queue is empty fill it with the new frame. 
    If the queue is populated, insert the new frame and pop the latest one
    Return both a list and numpy array of the frame queue
    '''
    frame_stack = prev_stack
    
    while len(frame_stack)<size+1:
        frame_stack.append(frame)
    frame_stack = frame_stack[-size:]

    np_frame_stack = np.stack(frame_stack, axis=0)

    return frame_stack, np_frame_stack

def disc_rewards(rewards, gamma):
    '''
    Return expected rewards for each timestep by discounting future rewards by gamma 
    '''
    discounted_episode_rewards = np.zeros_like(rewards)
    cumulative_reward = 0.0
    for i in reversed(range(len(rewards))):
        cumulative_reward = cumulative_reward * gamma + rewards[i]
        discounted_episode_rewards[i] = cumulative_reward

    return discounted_episode_rewards

def norm_rewards(reward_list):
    '''
    Normalize rewards from multiple episodes by subtracting the bias for each timestep
    '''
    result = []

    maxlen = 0
    for np_array in reward_list: 
        maxlen = max(maxlen, len(np_array))
        result.append(np.zeros_like(np_array))

    for i in range(maxlen):
        b, n = 0, 0
        for np_array in reward_list:
            if i >= len(np_array): continue
            b += np_array[i]
            n += 1

        for j, np_array in enumerate(reward_list):
            if i >= len(np_array): continue
            result[j][i] = np_array[i] - b/n

    return result

class dotdict(dict):
    '''Implements a dot dictionary'''
    def __getattr__(self, name):
        return self[name]

    def fill_w_default(self):
        if "dropout" not in self:
            self.dropout = 0


# ep_rewards = np.split(rewards, np.where(rewards==-99)[0]+1)[:-1]    
# mean = np.mean(discounted_episode_rewards)
# std = np.std(discounted_episode_rewards)
# discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)


# fig, ax = plt.subplots(nrows=2, ncols=3)
# for i, row in enumerate(ax):
#     for j, col in enumerate(row):
#         col.imshow(np_frame_stack[i*2+j])
# #print(np_frame_stack.shape)
# plt.show()

# fig, ax = plt.subplots(nrows=2, ncols=3)
# for i, row in enumerate(ax):
#     for j, col in enumerate(row):
#         col.imshow(s[0, i*2+j].cpu())

# plt.show()

#print(normalized_rewards)
# fig, ax = plt.subplots(nrows=2, ncols=3)
# for i, row in enumerate(ax):
#     for j, col in enumerate(row):
#         col.imshow(states[5, i*2+j])
# #print(np_frame_stack.shape)
# print(actions[5])
# print(normalized_rewards[5])
# plt.show()