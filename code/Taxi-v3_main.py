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

class AgentAssessment:
    def __init__(self):
        self.env = gym.make("Taxi-v3")

        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.9

        self.episodes = 1000
        self.ep_step= 100
        self.max_steps = 2500

        self.q_agent = Agent_QL(self.env)
        self.sarsa_agent = Agent_Sarsa(self.env)
        self.dqn_agent = DQN_Agent(self.env)

    def train_models(self):
        self.q_agent.train_agent(alpha = self.alpha, gamma = self.gamma, epsilon = self.epsilon, episodes = self.episodes, ep_step=self.ep_step, max_steps = self.max_steps, filename='files/training_ql.npy')
        self.sarsa_agent.train_agent(alpha = self.alpha, gamma = self.gamma, epsilon = self.epsilon, episodes = self.episodes, ep_step=self.ep_step, max_steps = self.max_steps, filename='files/training_sarsa.npy')
        self.dqn_agent.train_agent(episodes = self.episodes, ep_step=self.ep_step, filename = 'files/dqn_weights.h5f')

    def make_patch_spines_invisible(self,ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    def show_diagrams(self, ep, data1, data2, data3, mode):
        fig, host = plt.subplots()
        par1 = host.twinx()
        par2 = host.twinx()

        par2.spines["right"].set_position(("axes", 1.2))
        self.make_patch_spines_invisible(par2)
        par2.spines["right"].set_visible(True)

        p1, = host.plot(ep, data1, "b-", label="q")
        p2, = par1.plot(ep, data2, "r-", label="sarsa")
        p3, = par2.plot(ep, data3, "g-", label="dqn")

        host.set_xlabel("episodes")

        if mode == 1:
            host.set_ylabel("reward")
        else:
            host.set_ylabel("timesteps")

        host.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p2.get_color())
        par2.yaxis.label.set_color(p3.get_color())
        tkw = dict(size=4, width=1.5)
        host.tick_params(axis='y', colors=p1.get_color(), **tkw)
        par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
        par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
        host.tick_params(axis='x', **tkw)
        lines = [p1, p2, p3]
        host.legend(lines, [l.get_label() for l in lines])
        plt.show()

    def assess_models(self):
        stats = dict()

        ep = np.arange(1, self.episodes, self.ep_step)

        self.train_models()
        stats['q'] = self.q_agent.getStats()
        stats['sarsa'] = self.sarsa_agent.getStats()
        stats['dqn'] = self.dqn_agent.getStats()

        # plt.plot(ep, stats['sarsa']['reward'])
        # plt.show()
        self.show_diagrams(ep,stats['q']['reward'],stats['sarsa']['reward'],stats['dqn']['reward'],1)
        self.show_diagrams(ep,stats['q']['timesteps'],stats['sarsa']['timesteps'],stats['dqn']['timesteps'],2)

    
        print(f"Results for {self.episodes} episodes:")

        print("NO learning")
        brute_avg_timesteps, brute_avg_reward = brute_agent(self.env, self.episodes)
        print(f"Average timesteps per episode: {brute_avg_timesteps}")
        print(f"Average reward per episode: {brute_avg_reward}")

        print("Q learning")
        print(f"Average timesteps per episode: {mean(stats['q']['timesteps'])}")
        print(f"Average reward per episode: {mean(stats['q']['reward'])}")

        print("Sarsa learning")
        print(f"Average timesteps per episode: {mean(stats['sarsa']['timesteps'])}")
        print(f"Average reward per episode: {mean(stats['sarsa']['reward'])}")

        print("DQN learning")
        print(f"Average timesteps per episode: {mean(stats['dqn']['timesteps'])}")
        print(f"Average reward per episode: {mean(stats['sarsa']['reward'])}")

    def test(self):
        self.q_agent.simulate(filename='training_ql.npy', visualize = True, episodes=1)
        
agent = AgentAssessment()
# agent.test()
agent.assess_models()
