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
sim_episodes = 100

brute_avg_timesteps, brute_avg_reward = brute_agent(env,sim_episodes)

q_agent = Agent_QL(env)
q_agent.train_agent(alpha = 0.4, gamma = 0.95, epsilon = 0.9, episodes = 10000,filename='training_ql.npy')
q_avg_timesteps, q_avg_reward  = q_agent.simulate('training_ql.npy', visualize = False, episodes=sim_episodes)

sarsa_agent = Agent_Sarsa(env)
sarsa_agent.train_agent(alpha = 0.4, gamma = 0.999, epsilon = 0.9, episodes = 10000, max_steps = 2500, n_tests = 20,filename='training_sarsa.npy')
s_avg_timesteps, s_avg_reward  = sarsa_agent.simulate(filename = 'training_sarsa.npy', visualize = False, episodes=sim_episodes)

dqn_agent = DQN_Agent(env)
dqn_agent.train_agent('dqn_weights.h5f')
s_avg_timesteps, s_avg_reward = dqn_agent.simulate(episodes=sim_episodes, visualze = False)

print(f"Results for {sim_episodes} episodes:")

print("NO learning")
print(f"Average timesteps per episode: {brute_avg_timesteps}")
print(f"Average reward per episode: {brute_avg_reward}")

print("Q learning")
print(f"Average timesteps per episode: {q_avg_timesteps}")
print(f"Average reward per episode: {q_avg_reward }")

print("Sarsa learning")
print(f"Average timesteps per episode: {s_avg_timesteps}")
print(f"Average reward per episode: {s_avg_reward }")

print("DQN learning")
print(f"Average timesteps per episode: {s_avg_timesteps}")
print(f"Average reward per episode: {s_avg_reward}")

