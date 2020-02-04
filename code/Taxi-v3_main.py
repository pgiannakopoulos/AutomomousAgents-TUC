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
from brute_algorithm import brute_agent

class AgentAssessment:
    def __init__(self):
        # Set parameters
        self.env = gym.make("Taxi-v3")
        self.episodes = 20000
        self.ep_step=  100
        self.max_steps = 2500

        # Initialize the agents
        self.q_agent = Agent_QL(self.env)
        self.sarsa_agent = Agent_Sarsa(self.env)
        self.dqn_agent = DQN_Agent(self.env)

    def train_models(self):
        self.q_agent.train_agent(alpha = 0.75, gamma = 0.9, epsilon = 0.9, episodes = self.episodes, ep_step=self.ep_step, max_steps = self.max_steps, filename='training/training_ql.npy')
        self.sarsa_agent.train_agent(alpha = 0.5, gamma = 0.975, epsilon = 0.9, episodes = self.episodes, ep_step=self.ep_step, max_steps = self.max_steps, filename='training/training_sarsa.npy')
        self.dqn_agent.train_agent(episodes = self.episodes, ep_step=self.ep_step, filename = 'training/dqn_weights.h5f')

    # Needed for show_diagramms function
    def make_patch_spines_invisible(self,ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    # Show plots with assessment results
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
        plt.title("Average {}".format(type))
        plt.savefig("img/{}_bars.png".format(type))   # save the figure to file
        plt.close() 
        
    def assess_models(self):
        stats = dict()

        ep = np.arange(1, self.episodes, self.ep_step)

        if not os.listdir('img'):   #if folder is empty
            if os.listdir('training'):
                print("*In order to continue in algorithms assessment delete or rename the files in training folder.*")
                return

            # Get the data for assessment
            self.train_models()
            stats['q'] = self.q_agent.getStats()          #lists with data
            stats['sarsa'] = self.sarsa_agent.getStats()
            stats['dqn'] = self.dqn_agent.getStats()

            # Plot results
            self.show_diagrams(ep,stats['q']['reward'],stats['sarsa']['reward'],stats['dqn']['reward'],1)
            self.show_diagrams(ep,stats['q']['timesteps'],stats['sarsa']['timesteps'],stats['dqn']['timesteps'],2)    
            
            # Plot average data after training
            q_time, q_rew = self.q_agent.simulate(filename='training/training_ql.npy', visualize = False, episodes=100)
            sarsa_time, sarsa_rew = self.sarsa_agent.simulate(filename='training/training_sarsa.npy', visualize = False, episodes=100)
            dqn_time, dqn_rew = self.dqn_agent.simulate(visualize=False, episodes=100)

            data1 = [q_rew, sarsa_rew, dqn_rew]
            self.show_bars(data1, 1)

            data2 = [q_time, sarsa_time, dqn_time]
            self.show_bars(data2, 2)

            # data = [stats['q']['reward'][-1],stats['sarsa']['reward'][-1],stats['dqn']['reward'][-1]]
            # self.show_bars(data, 1)

            # data = [stats['q']['timesteps'][-1],stats['sarsa']['timesteps'][-1],stats['dqn']['timesteps'][-1]]
            # self.show_bars(data, 2)

            print(f"Results for {self.episodes} episodes in folder img!")

            # Save the arrays to files for further analysis
            np.save('plot_data/q_rewards.npy', stats['q']['reward'])
            np.save('plot_data/q_timesteps.npy', stats['q']['timesteps'])
            np.save('plot_data/q_rewards.npy', stats['sarsa']['reward'])
            np.save('plot_data/q_timesteps.npy', stats['sarsa']['timesteps'])
            np.save('plot_data/q_rewards.npy', stats['dqn']['reward'])
            np.save('plot_data/q_timesteps.npy', stats['dqn']['timesteps'])
            np.save('plot_data/episodes.npy', ep)
        else:
            print("*Plot images already exist in folder img*")

        print("NO learning")
        brute_avg_timesteps, brute_avg_reward = brute_agent(self.env, self.episodes)
        print(f"Average timesteps per episode: {brute_avg_timesteps}")
        print(f"Average reward per episode: {brute_avg_reward}")

    # Visualize the agent simulation
    def simulate_agent(self, type):
        if type == 1:
            self.q_agent.simulate(filename='training/training_ql.npy', visualize = True, episodes=1)
        elif type == 2:
            self.sarsa_agent.simulate(filename='training/training_sarsa.npy', visualize = True, episodes=1)
        elif type == 3:
            self.dqn_agent.loadWeights(filename = 'training/dqn_weights.h5f')
            self.dqn_agent.simulate(visualize=True, episodes=1)

    # Search for optimal hyperparameters
    def find_hyperparameters(self, algor):
        if (algor == 1):
            agent = self.q_agent
            name = "q_learning"
        else:
            agent = self.sarsa_agent
            name = "sarsa_learning"

        alpha_table = np.linspace(start = .5, stop = 1, num = 5)
        gamma_table = np.linspace(start = .9, stop = 1, num = 5)
        epsilon_table = np.linspace(start = .4, stop = .9, num = 5)
        episodes = 1000

        best_value = float("-inf")
        best_alpha = 0
        best_gamma = 0
        best_epsilon = 0

        for alpha in alpha_table:
            for gamma in gamma_table:
                for epsilon in epsilon_table:
                    agent.train_agent(alpha = alpha, gamma = gamma, epsilon = epsilon, episodes = episodes, ep_step=self.ep_step, max_steps = self.max_steps, filename="hyperparameters/"+name+'.npy')
                    timesteps, rewards = agent.simulate(filename="hyperparameters/"+name+'.npy', visualize = False, episodes=100)
                    ratio = rewards / timesteps

                    #Check if a better combination has been found
                    if (ratio > best_value):
                        best_value = ratio
                        best_alpha = alpha
                        best_gamma = gamma
                        best_epsilon = epsilon
                    os.remove("hyperparameters/"+name+'.npy')
                    print("alpha:{} , gamma:{}, epsilon:{} -- ratio:{}".format(alpha, gamma, epsilon, ratio))
        print("{} ==> alpha:{} , gamma:{}, epsilon:{} with ratio:{}".format(name,best_alpha, best_gamma, best_epsilon, best_value))
        return best_alpha, best_gamma, best_epsilon

    def menu(self):
        print("Welcome to Taxi-v3 with Q, SARSA, DQN learning!")
        print("=====>  MENU  <=====")
        print("--> 1: Train & Assess the algorithms.")
        print("--> 2: Simulate.")
        print("--> 3: Find the optimal hyperparameters.")
        print("--> To exit press any number.")
        print("*1 & 3 will take some time.*")
        print("Choose one option by typing the corresponding number.")
        try:
            option=int(input('Input: '))
            if option == 1:
                self.assess_models()
            elif option == 2:
                print('Type 1 for Q, 2 for SARSA, 3 for DQN:')
                try:
                    option=int(input('Input: '))
                    if option > 3 and option < 0:
                        print("Invalid option!")
                        return
                    self.simulate_agent(option)
                except ValueError:
                    print("Not a number")
                    return
            elif option == 3:
                print('Type 1 for Q, 2 for SARSA:')
                try:
                    option=int(input('Input: '))
                    if option > 2 and option < 0:
                        print("Invalid option!")
                        return
                    self.find_hyperparameters(option)
                except ValueError:
                    print("Not a number")
                    return
            else:
                print("Bye!")
        except ValueError:
            print("Not a number")

agent = AgentAssessment()
agent.menu()
# agent.assess_models()
# agent.simulate_agent(3)

#q_learning ==> alpha:0.75 , gamma:0.9, epsilon:0.9 with ratio:0.6881028938906754
# agent.find_hyperparameters(1)

#sarsa_learning ==> alpha:0.5 , gamma:0.975, epsilon:0.9 with ratio:0.2330960854092527
# agent.find_hyperparameters(2)