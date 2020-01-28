import gym
import numpy as np
import random
from IPython.display import clear_output
from time import sleep
import os.path
from os import path
from q_learning import Agent_QL
from sarsa_learning import Agent_Sarsa
from deep_q_learning import DQN_Agent
from brute_algorithm import brute_agent

env = gym.make("Taxi-v3")
env.reset()

# q_agent = Agent_QL(env)
# alpha = 0.1
# gamma = 0.6
# epsilon = 0.1
# episodes = 10000
# q_agent.train_agent(alpha, gamma, epsilon, episodes,'training_ql.npy')
# q_agent.simulate('training_ql.npy')


# sarsa_agent = Agent_Sarsa(env)
# alpha = 0.4
# gamma = 0.999
# epsilon = 0.9
# episodes = 3000
# max_steps = 2500
# n_tests = 20
# sarsa_agent.train_agent(alpha, gamma, epsilon, episodes, max_steps, n_tests,'training_sarsa.npy')
# sarsa_agent.simulate()

dqn_agent = DQN_Agent(env)
dqn_agent.train_agent('dqn_weights.h5f')
dqn_agent.simulate(episodes=50000)


# brute_agent = brute_agent(env)

