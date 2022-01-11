import sys
import os
sys.path.append(os.getcwd())

import gym
import torch

from agent import Agent
from model import DQN
#from config import Config

env = gym.make("Taxi-v3").env
env.reset()
import argparse
import os

import yaml

if os.path.exists("config.yaml"):
    # If config.yml is provided, always use that.
    #config = yaml.load(open("config.yaml"))
    config = yaml.safe_load(open("config.yaml"))
elif os.path.exists("config.yml.in"):
    # If config.yml.in is provided, use it as defaults with CLI
    # overrides.
    config = yaml.load(open("config.yml.in"))
    assert isinstance(config, dict), config
    p = argparse.ArgumentParser()
    for name, default in sorted(config.items()):
        p.add_argument("--%s" % name, default=default, type=type(default))
        args = p.parse_args()
        config.update(dict(args._get_kwargs()))
else:
    assert False, "missing config: expected config.yml or config.yml.in"

print("Using config: %s" % config) 
input()

class Config:
    class training:
        batch_size = batch_size
        learning_rate = learning_rate
        loss = loss
        num_episodes = num_episodes
        train_steps = train_steps
        warmup_episode = warmup_episode
        save_freq = save_freq
    
    class optimizer:
        name = name
        lr_min = lr_min
        lr_decay = lr_decay
    
    class rl:
        gamma = gamma
        max_steps_per_episode = max_steps_per_episode
        target_model_update_freq = target_model_update_freq
        memory_capacity = memory_capacity
        num_episodes = num_episodes
    
    class epsilon:
        max_epsilon = max_epsilon
        min_epsilon = min_epsilon
        decay_epsilon = decay_epsilon

class Config_old:
    class training:
        batch_size = 128
        learning_rate = 0.001
        loss = "huber"
        num_episodes = 10000
        train_steps = 1000000
        warmup_episode = 100
        save_freq = 1000
    
    class optimizer:
        name = "adam"
        lr_min = 0.0001
        lr_decay = 5000
    
    class rl:
        gamma = 0.99
        max_steps_per_episode = 200
        target_model_update_freq = 20
        memory_capacity = 50000
        num_episodes = 20
    
    class epsilon:
        max_epsilon = 1
        min_epsilon = 0.1
        decay_epsilon = 600
config = Config()


# Config flags for GuildAi mlops.
# Will overwrite what's in config.py

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(THIS_FOLDER, 'foo')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DQN
#load_model=True
load_model=False
if load_model:
    weights = torch.load(path)
    #model.load_state_dict(torch.load(path))
    model.load_state_dict(weights)
    model.eval()

agent = Agent(env, config, model, device)
agent.compile()
agent.fit(path,verbose=True)
#agent.play(verbose=True,sleep=0.1,max_steps=100)

#agent.fit(path,verbose=False)
#agent.play(verbose=False,sleep=0.0,max_steps=100)
#agent.play(verbose=True,sleep=0.1,max_steps=50)
