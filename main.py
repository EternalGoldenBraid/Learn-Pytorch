import sys
import os
sys.path.append(os.getcwd())

import gym

from agent import Agent
from model import DQN
from config import Config

env = gym.make("Taxi-v3").env
env.reset()

config = Config()

agent = Agent(env, config, DQN)
agent.compile()
agent.fit(verbose=True)
