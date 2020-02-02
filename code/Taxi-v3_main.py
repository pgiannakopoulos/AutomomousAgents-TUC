import gym
import numpy as np
from statistics import mean
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep
import os
from os import path

from q_learning import Agent_QL
from sarsa_learning import Agent_Sarsa
from deep_q_learning import DQN_Agent
from deepsarsa import DSARSA_Agent
from brute_algorithm import brute_agent

class AgentAssessment:
    def __init__(self):
        self.env = gym.make("Taxi-v3")

        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.9

        self.episodes = 10000
        self.ep_step= 100
        self.max_steps = 2500

        self.q_agent = Agent_QL(self.env)
        self.sarsa_agent = Agent_Sarsa(self.env)
        self.dqn_agent = DQN_Agent(self.env)
        self.dsarsa_agent = DSARSA_Agent(self.env)

    def train_models(self):
        self.q_agent.train_agent(alpha = self.alpha, gamma = self.gamma, epsilon = self.epsilon, episodes = self.episodes, ep_step=self.ep_step, max_steps = self.max_steps, filename='training/training_ql.npy')
        self.sarsa_agent.train_agent(alpha = self.alpha, gamma = self.gamma, epsilon = self.epsilon, episodes = self.episodes, ep_step=self.ep_step, max_steps = self.max_steps, filename='training/training_sarsa.npy')
        self.dqn_agent.train_agent(episodes = self.episodes, ep_step=self.ep_step, filename = 'training/dqn_weights.h5f')

    def make_patch_spines_invisible(self,ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    def show_diagrams(self, ep, data1, data2, data3, mode):
        if mode == 1:
            type = 'Reward'
        else:
            type = 'Timesteps'

        fig, host = plt.subplots()
        par1 = host.twinx()
        par2 = host.twinx()

        par2.spines["right"].set_position(("axes", 1.2))
        self.make_patch_spines_invisible(par2)
        par2.spines["right"].set_visible(True)

        p1, = host.plot(ep, data1, "b-", label="q")
        p2, = par1.plot(ep, data2, "r-", label="sarsa")
        p3, = par2.plot(ep, data3, "g-", label="dqn")

        plt.title("{} Diagram".format(type))
        host.set_xlabel("episodes")
        host.set_ylabel(str(type))
    
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
        fig.savefig("img/{}_diagram.png".format(type))   # save the figure to file
        plt.close(fig) 

    def show_bars(self, data, mode):
        if mode == 1:
            type = 'Reward'
        else:
            type = 'Timesteps'

        objects = ('Q', 'SARSA', 'DQN')

        y_pos = np.arange(len(objects))

        plt.bar(y_pos, data, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel("{} Diagram".format(type))
        plt.title("Average {} per episode".format(type))
        plt.savefig("img/{}_bars.png".format(type))   # save the figure to file
        plt.close() 
        
    def assess_models(self):
        stats = dict()

        ep = np.arange(1, self.episodes, self.ep_step)

        if not os.listdir('img'):   #if folder is empty
            if os.listdir('training'):
                print("*In order to continue in algorithms assessment delete or rename the files in training folder.*")
                return

            self.train_models()
            stats['q'] = self.q_agent.getStats()
            stats['sarsa'] = self.sarsa_agent.getStats()
            stats['dqn'] = self.dqn_agent.getStats()

            # Plot results
            self.show_diagrams(ep,stats['q']['reward'],stats['sarsa']['reward'],stats['dqn']['reward'],1)
            self.show_diagrams(ep,stats['q']['timesteps'],stats['sarsa']['timesteps'],stats['dqn']['timesteps'],2)    
            
            # Plot average data

            data = [mean(stats['q']['reward']),mean(stats['sarsa']['reward']),mean(stats['dqn']['reward'])]
            self.show_bars(data, 1)

            data = [mean(stats['q']['timesteps']),mean(stats['sarsa']['timesteps']),mean(stats['dqn']['timesteps'])]
            self.show_bars(data, 2)

            print(f"Results for {self.episodes} episodes in folder img!")
        else:
            print("*Plot images already exist in folder img*")

        print("NO learning")
        brute_avg_timesteps, brute_avg_reward = brute_agent(self.env, self.episodes)
        print(f"Average timesteps per episode: {brute_avg_timesteps}")
        print(f"Average reward per episode: {brute_avg_reward}")

    def simulate_agent(self, type):
        if type == 1:
            self.q_agent.simulate(filename='training/training_ql.npy', visualize = True, episodes=1)
        elif type == 2:
            self.sarsa_agent.simulate(filename='training/training_sarsa.npy', visualize = True, episodes=1)
        elif type == 3:
            self.dqn_agent.simulate(visualize=True, episodes=1)
            
        
agent = AgentAssessment()
agent.assess_models()
# agent.simulate_agent(3)
