"""
Actions:
There are 6 discrete deterministic actions:
- 0: move south
- 1: move north
- 2: move east
- 3: move west
- 4: pickup passenger
- 5: drop off passenger
Rewards:
There is a default per-step reward of -1,
except for delivering the passenger, which is +20,
or executing "pickup" and "drop-off" actions illegally, which is -10.
"""

import gym
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import namedtuple, deque
from numpy.random import default_rng
from IPython.display import clear_output
from time import sleep
from os import system

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

SEED = 42

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment
#env = gym.make("Taxi-v3") # Step limit == 200
env = gym.make("Taxi-v3").env
env.reset()
#env.render()

clear = lambda: system('clear')
def print_frames(frames):
    
    for i, frame in enumerate(frames):
        #clear_output(wait=True)
        clear()
        #print(frame['frame'].getvalue())
        print(frame['frame'])
        print(f"Timestep: {i+1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)


def to_numpy_2D(list_2d):
    """
    Padd a python list so that it fits the shape of a 2d nimpy array.
    """

    max_col = max((len(el) for el in list_2d))
    max_row = len(list_2d)

    for row in list_2d:
        row.extend([0]*(max_col-len(row)))
        #row.extend([np.nan]*(max_col-len(row)))

    list_2d = np.asarray(list_2d)
    return list_2d

class ReplayMemory(object):
# Memory of transitions that agent observes.

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """ Save a transition """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent:
    """
    From source code: https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
    There are 500 discrete states since there are 25 taxi positions, 
    5 possible locations of the passenger (including the case when the passenger is in the taxi), 
    and 4 destination locations.
    Note that there are 400 states that can actually be reached during an episode. 
    The missing states correspond to situations in which the passenger is at the same location as their destination,
    as this typically signals the end of an episode.
    Four additional states can be observed right after a successful episodes, when both the passenger and the taxi are at the destination.
                This gives a total of 404 reachable discrete states.
    """
    def __init__(self, envm, epsilon):
        self.seed=SEED
        self.rng = default_rng(SEED)
        self.number_states = env.observation_space.n
        self.number_actions = env.action_space.n
        #self.Qtable=np.zeros((self.number_states,self.number_actions))
        self.Qtable=self.rng.random((self.number_states,self.number_actions))
        self.epsilon = epsilon
        self.epsilon_min = self.epsilon*1e-4
        self.step_count = 0

    def action(self, state, step, operation):
        """
        Choose the next action based on the current Q-table.
        Return a random integer for inital state
        """

        if operation == 'training' and random.uniform(0,1) < self.epsilon:
            action = env.action_space.sample()
            #return random.randrange(0,self.number_actions)
        else:
            action = np.argmax(self.Qtable[state])

        self.step_count += 1

        #if  and self.epsilon >= self.epsilon_min:
        if self.step_count % 100 == 0 and self.epsilon >= self.epsilon_min:
            self.epsilon -= self.epsilon/1000
        return action

    def update(self, state, new_state, reward, action):
        """
        Update the Qtable after an action results in new state and reward.
        """

        old_value = self.Qtable[state][action]
        self.Qtable[state][action] = old_value + alpha*(reward+gamma*np.max(self.Qtable[new_state]) - old_value)
        #self.Qtable[state][action] = (1-alpha)*old_value + alpha*(reward+gamma*np.max(self.Qtable[new_state]))

def navigate(agent, alpha, gamma, epochs, operation, steps_max):

    # For animation and not flooding terminal.
    frames = []

    # Measures for success
    # All rewards, actions, penalties and steps over all steps and epochs
    rewards = [] 
    total_actions = []
    penalties = []
    steps = np.zeros((epochs,1))
    dropoffs = np.zeros((epochs,1)) # Succesful dropoffs

    steps_max = steps_max
    step = 0
    epoch = 0
    while epoch < epochs:

        frames.append([])
        rewards.append([])
        total_actions.append([])
        penalties.append([])

        done = False
        reward = 0
        penalty = 0
        state = env.reset()
        #while not done:
        while step < steps_max and not done:
            step += 1
            action = agent.action(state,step,operation)
            new_state, reward, done, info = env.step(int(action))
            if operation == 'training': 
                agent.update(state, new_state, reward, action)
            state = new_state

            # Collect step results
            if reward == -10:
                penalties[epoch].append(1)
            else:
                penalties[epoch].append(0)
            rewards[epoch].append(reward)
            total_actions[epoch].append(action)
            frames[epoch].append({
                'frame': env.render(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward,
                })


            # Drop off succesful
            if reward == 20:
                #print("Sucessful dropoff! ", end="")
                dropoffs[epoch] += 1

        steps[epoch] = step
        penalty_epoch = sum(penalties[epoch])
        print(f"{operation} epoch: {epoch} - Steps: {step} - penalties: {penalty_epoch}")
        epoch += 1
        step=0
    
    return total_actions, rewards, frames, penalties, steps, dropoffs

""" Training setup """
alpha = 0.35 # Learning rate
gamma = 0.99 # Discount factor, greedy (0) vs long term (1)
epochs_training = 10000
steps_max_training = 1000
epsilon = 0.53 # Improve to be proportional to nubmer of steps and epochs taken?

#clear_output(wait=True)
#clear()
agent = Agent(env, epsilon)
operation = 'training'
outputs = []
plt.ion()
outputs = navigate(agent, alpha, gamma, epochs_training, operation=operation, steps_max=steps_max_training)
actions_training    = outputs[0]
rewards_training    = outputs[1]
frames_training     = outputs[2]
penalties_training  = outputs[3]
steps_training      = outputs[4]
dropoffs_training   = outputs[5]

""" Testing setup """
epochs_testing = 10
steps_max_testing = 200
operation = 'testing'
outputs = navigate(agent, alpha, gamma, epochs_testing, operation=operation, steps_max=steps_max_testing)
actions_testing    = outputs[0]
rewards_testing    = outputs[1]
frames_testing     = outputs[2]
penalties_testing  = outputs[3]
steps_testing      = outputs[4]
dropoffs_testing   = outputs[5]

""" Evaluation """
### Training ###
# Actions 
actions_training = to_numpy_2D(actions_training)
action_training_epoch = actions_training.mean(axis=1)
action_training_mean = np.mean(action_training_epoch)
# Steps
steps_training_mean = np.array(steps_training).mean(axis=0)
# Reward
rewards_training = to_numpy_2D(rewards_training)
reward_training_epoch = rewards_training.sum(axis=1)
reward_training_mean = np.mean(reward_training_epoch)
reward_training_mean_per_step = np.mean(reward_training_epoch/steps_training)
# Penalty
penalties_training = to_numpy_2D(penalties_training)
penalties_training_epoch = penalties_training.sum(axis=1)
penalties_training_mean = np.mean(penalties_training_epoch)
# Dropoffs
dropoffs_training_mean = np.array(dropoffs_training).mean()
print(f"""
Training Summary:
- Average reward per epoch: {reward_training_mean:.2f},
- Average reward per step: {reward_training_mean_per_step:.2f},
- Average penalty: {penalties_training_mean:.2f}
- Average number of actions: {steps_training_mean[0]:.2f},
- Average action: {action_training_mean:.2f},
- Average number of succesful dropoffs: {dropoffs_training_mean:.2f},
""")

### Testing ###
# Actions 
actions_testing = to_numpy_2D(actions_testing)
action_testing_epoch = actions_testing.mean(axis=1)
action_testing_mean = np.mean(action_testing_epoch)
# Steps
steps_testing_mean = np.array(steps_testing).mean(axis=0)
# Reward
rewards_testing = to_numpy_2D(rewards_testing)
reward_testing_epoch = rewards_testing.sum(axis=1)
reward_testing_mean = np.mean(reward_testing_epoch)
reward_testing_mean_per_step = np.mean(reward_testing_epoch/steps_testing)
# Penalty
penalties_testing= to_numpy_2D(penalties_testing)
penalties_testing_epoch = penalties_testing.sum(axis=1)
penalties_testing_mean = np.mean(penalties_testing_epoch)
# Dropoffs
dropoffs_testing_mean = np.array(dropoffs_testing).mean()
print(f"""
Testing Summary:
c Average reward per epoch: {reward_testing_mean:.2f},
- Average reward per step: {reward_testing_mean_per_step:.2f},
- Average penalty: {penalties_testing_mean:.2f}
c Average number of actions: {steps_testing_mean[0]:.2f},
- Average action: {action_testing_mean:.2f},
- Acerage number of succesful dropoffs: {dropoffs_testing_mean:.2f},
""")

""" Visualization """
# Rewards
fig1, axes_rw = plt.subplots(nrows=1,ncols=2)
ax_rw = axes_rw.ravel()
ax_rw[0].plot(range(epochs_training),reward_training_epoch, label="Reward")
ax_rw[0].set_title("Average reward per training epoch")
ax_rw[1].plot(range(epochs_testing),reward_testing_epoch, label="Reward")
ax_rw[1].set_title("Average reward per testing epoch")
# Penalty
fig2, axes_penalty = plt.subplots(nrows=1,ncols=2)
ax_penalty = axes_penalty.ravel()
ax_penalty[0].plot(range(epochs_training),penalties_training_epoch, label="Penalties")
ax_penalty[0].set_title("Average penalty per training epoch")
ax_penalty[1].plot(range(epochs_testing),penalties_testing_epoch, label="Penalties")
ax_penalty[1].set_title("Average penalty per testing epoch")
# Sucesful Dropoffs
fig3, axes_drp = plt.subplots(nrows=1,ncols=2)
ax_drp = axes_drp.ravel()
#ax_drp[0].scatter(range(epochs_training),dropoffs_training, label="Dropoffs")
ax_drp[0].hist(dropoffs_training,len(dropoffs_training), label="Dropoffs")
ax_drp[0].set_title("Correct dropoffs per training epoch")
#ax_drp[1].scatter(range(epochs_testing),dropoffs_testing, label="Dropoffs")
ax_drp[1].hist(dropoffs_testing,len(dropoffs_testing), label="Dropoffs")
ax_drp[1].set_title("Correct dropoffs per testing epoch")
#plt.show()
fig1.savefig("rewards.png")
fig2.savefig("penalties.png")
fig3.savefig("dropoffs.png")
plt.close(fig1)
plt.close(fig2)
plt.close(fig3)
plt.show()

q = input("Watch frames? (Y/n): ")
if q == 'Y' or q == 'y':
    frames_epoch = frames_testing
    for epoch, frames in enumerate(frames_epoch):
        for frame in frames:
            if frame['reward'] == 20:
                print_frames(frames)
