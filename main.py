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
    print(os.getcwd())
    print(os.listdir())
    assert False, "missing config: expected config.yaml or config.yml.in"

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
