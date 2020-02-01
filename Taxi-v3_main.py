import gym
import numpy as np
from statistics import mean
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep
import os.path
from os import path

from q_learning import Agent_QL
from sarsa_learning import Agent_Sarsa
from deep_q_learning import DQN_Agent
from brute_algorithm import brute_agent

env = gym.make("Taxi-v3")

# Create Models
q_agent = Agent_QL(env)
q_agent.train_agent(alpha = 0.1, gamma = 0.95, epsilon = 0.9, episodes = 10000, max_steps = 2500, filename='training_ql.npy')

sarsa_agent = Agent_Sarsa(env)
sarsa_agent.train_agent(alpha = 0.1, gamma = 0.95, epsilon = 0.9, episodes = 10000, max_steps = 2500, filename='training_sarsa.npy')

dqn_agent = DQN_Agent(env)
dqn_agent.train_agent('dqn_weights.h5f')


# Simulation

stats = dict()
stats['brute'] = list()
stats['q'] = list()
stats['sarsa'] = list()
stats['dqn'] = list()

sim_max = 10
sim_step = 1
for sim_episodes in range(1,sim_max,sim_step):
    brute_avg_timesteps, brute_avg_reward = brute_agent(env,sim_episodes)
    q_avg_timesteps, q_avg_reward  = q_agent.simulate('training_ql.npy', visualize = False, episodes=sim_episodes)
    s_avg_timesteps, s_avg_reward  = sarsa_agent.simulate(filename = 'training_sarsa.npy', visualize = False, episodes=sim_episodes)
    dqn_avg_timesteps, dqn_avg_reward = dqn_agent.simulate(episodes=sim_episodes, visualze = False)

    stats['brute'].append(brute_avg_timesteps)
    stats['q'].append(q_avg_timesteps)
    stats['sarsa'].append(s_avg_timesteps)
    stats['dqn'].append(dqn_avg_timesteps)

# Plot Data
# x = np.arange(1,sim_max,sim_step)

# def make_patch_spines_invisible(ax):
#     ax.set_frame_on(True)
#     ax.patch.set_visible(False)
#     for sp in ax.spines.values():
#         sp.set_visible(False)

# fig, host = plt.subplots()
# par1 = host.twinx()
# par2 = host.twinx()

# par2.spines["right"].set_position(("axes", 1.2))
# make_patch_spines_invisible(par2)
# par2.spines["right"].set_visible(True)

# p1, = host.plot(x, stats['q'], "b-", label="q")
# p2, = par1.plot(x, stats['sarsa'], "r-", label="sarsa")
# p3, = par2.plot(x, stats['dqn'], "g-", label="dqn")

# host.set_xlabel("steps")
# host.set_ylabel("reward")
# host.yaxis.label.set_color(p1.get_color())
# par1.yaxis.label.set_color(p2.get_color())
# par2.yaxis.label.set_color(p3.get_color())
# tkw = dict(size=4, width=1.5)
# host.tick_params(axis='y', colors=p1.get_color(), **tkw)
# par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
# par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
# host.tick_params(axis='x', **tkw)
# lines = [p1, p2, p3]
# host.legend(lines, [l.get_label() for l in lines])
# plt.show()



print(f"Results for {sim_episodes} episodes:")

print("NO learning")
print(f"Average timesteps per episode: {mean(stats['brute'])}")
print(f"Average reward per episode: {brute_avg_reward}")

print("Q learning")
print(f"Average timesteps per episode: {mean(stats['q'])}")
print(f"Average reward per episode: {q_avg_reward }")

print("Sarsa learning")
print(f"Average timesteps per episode: {mean(stats['sarsa'])}")
print(f"Average reward per episode: {s_avg_reward }")

print("DQN learning")
print(f"Average timesteps per episode: {mean(stats['dqn'])}")
print(f"Average reward per episode: {dqn_avg_timesteps}")

