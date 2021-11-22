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
from itertools import count

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

class DQN(nn.module):

    #def __init__(self,h,w, outputs):
    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.emb = nn.Embedding(500,4)
        self.l1 = nn.Linear(4,50)
        self.l2 = nn.Linear(50,50)
        self.l3 = nn.Linear(50, outputs)

        # For vision version.
        #self.conv1 = nn.conv2d(500,12,kernel_size=5, stride=2)
        #self.bn1 = nn.BatchNorm2d(12)
        #self.conv1 = nn.conv2d(12,6,kernel_size=5, stride=2)
        #self.bn1 = nn.BatchNorm2d(12)

        # Compute size of outputs of conv2d
        # CONFIRM THE MATH? WHAT AND WHY?
        #def conv2d_size_out(size, kernel_size=5, stride =2):
        #    return (size- (kernel_size-1) -1)//stride + 1
        #convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        #convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        #linear_input_size = convw*convh*32
        #self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.l1(self.emb(x)))
        x = F.relu(self.l2(self.l1(x)))
        x = self.l3(x)
        return x

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
    def __init__(self, env, config):
        self.seed=SEED
        self.rng = default_rng(SEED)
        self.number_states = env.observation_space.n
        self.number_actions = env.action_space.n
        #self.Qtable=self.rng.random((self.number_states,self.number_actions))
        self.config = Config
        self.step_count = 0
        self.device = device
        self.memory = None
        self.loss = None
        self.model = None

    def compile(self):
        """ Initialize the model """
        n_actions = self.number_actions

        self.model = DQN(n_actions).to(self.device)
        self.target_model = DQN(n_actions).to(self.device)
        self.target_model.eval() # Evaluation mode. train = false
        self.optimizer = optim.Adam(self.model.parameters(), 
                lr=self.config.training.learning_rate)
        #self.optimizer = optim.Adam(self.model.parameters(), 
                #lr=self.config["training"]["learning_rate"])

    def _get_epsilon(self,episode):
        eps_min = self.config.epsilon.epsilon_min
        eps_max = self.config.epsilon.epsilon_max
        eps_decay = self.config.epsilon.decay_epsilon
        epsilon = eps_min + (eps_max-eps_min)*np.exp(-episode/decay_epsilon)
                
    def _get_action(self, state):
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            predicted = self.model(torch.tensor([state], device=self.device))
            action = predicted.max(1)[1]
        return action.item()

    def _choose_action(self, state, epsilon):
        if self.rng.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = self._get_action(state)
        return action

    def _adjust_learning_rate(self, episode):
        # TODO
        if True: 
            a = None

    def _train_model(self):
        if len(self.memory) < self.config.training.batch_size:
            return
        transitions = self.memory.sample(self.config.training.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        # >>> zip(*[('a', 1), ('b', 2), ('c', 3)]) === zip(('a', 1), ('b', 2), ('c', 3))
        # [('a', 'b', 'c'), (1, 2, 3)]
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        # Compute prediced Q values - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        predicted_q_value = self.model(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute the expected Q values
        # t.max(1) will return largest column value of each row.
        # First column on max result is max element for each row.
        # Target networks perceived Q values for next states following training
        # networks "choice" of action.
        next_state_values = self.target_model(next_state_batch).max(1)[0]
        expected_q_values = (~done_batch * next_state_values * self.config.rl.gamma) + reward_batch
        #expected_q_values = (~done_batch * next_state_values * self.config["rl"]["gamma"]) + reward_batch

        # Compute Huber loss.
        loss_function = nn.SmoothL1Loss()
        loss = self.loss_function(predicted_q_value, expected_q_values.unsqueeze(1))

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clam_(-1,1)
        self.optimizer.step()


    def _update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _remember(self, state, action, next_state, reward, done):
        self.memory.push(torch.tensor([state], device=self.device),
                        torch.tensor([action], device=self.device, dtype=torch.long),
                        torch.tensor([next_state], device=self.device),
                        torch.tensor([reward], device=self.device),
                        torch.tensor([done], device=self.device, dtype=torch.bool))

    def fit(self):
        # Stateless counterpart for nn.SmoothL1Loss()
        self.loss = F.smooth_l1_loss
        self.memory= ReplayMemory(self.config.rl.max_queue_length)

        epsilon = 1

        for i_episode in range(self.config.num_episodes):
            state = self.env.reset()
            if i_episode >= self.config.training.warmup_episode:
                epsilon = self._get_epsilon(i_episode - self.config.training.warmup_episode)

            for step in count():
                action = _get_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)

                self._remember(state, action, next_state, reward, done)

                if i_episode >= self.config.training.warmup_episode:
                    self._train_model()
                    #self._adjust_learning_rate(i_episode - self.config.training.warmup_episode + 1) # TODO
                    done = (step == self.config.rl.max_steps_per_episode - 1) or done
                else:
                    done = (step == 5 * self.config.rl.max_steps_per_episode -1) or done # Justify choice of magic number!

        ## TO DO CONTINUE HERE

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
Transition = namedtuple('Transition',
    ('state', 'action', 'next_state', 'reward', 'done'))

class Config:
    class Training:
        batch_size = 128,
        learning_rate = 0.001,
        loss = "huber",
        num_episodes = 10000,
        train_steps = 1000000,
        warmup_episode = 10,
        save_freq = 1000,

    class optimizer:
        name = adam,
        lr_min = 0.0001,
        lr_decay = 5000,

    class rl:
        gamma = 0.99,
        max_steps_per_episode = 100,
        target_model_update_episodes = 20,
        max_queue_length = 50000,
    
    class epsilon:
        max_epsilon = 1,
        min_epsilon = 0.1,
        decay_epsilon = 400,

#clear_output(wait=True)
#clear()
agent = Agent(env, config)
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
