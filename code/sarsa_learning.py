import gym
import numpy as np
import random
from IPython.display import clear_output
from time import sleep
import os.path
from os import path

"""
SARSA on policy learning python implementation.
This is a python implementation of the SARSA algorithm in the Sutton and Barto's book on
RL. It's called SARSA because - (state, action, reward, state, action). The only difference
between SARSA and Qlearning is that SARSA takes the next action based on the current policy
while qlearning takes the action with maximum utility of next state.
Using the simplest gym self.environment for brevity: https://gym.openai.com/self.envs/FrozenLake-v0/
"""

class Agent_Sarsa:
    """docstring for ClassName"""
    def __init__(self, env, alpha_decay = 0.0001, gamma_decay = 0.0001, epsilon_decay = 0.0001):
        self.env = env

        self.Q = None
        self.alpha_decay = alpha_decay
        self.gamma_decay = gamma_decay
        self.epsilon_decay = epsilon_decay

        self.stats = dict()
        self.stats['timesteps'] = list()
        self.stats['reward'] = list()

    def epsilon_greedy(Q, epsilon, n_actions, s, train=False):
        """
        @param Q Q values state x action -> value
        @param epsilon for exploration
        @param s number of states
        @param train if true then no random actions selected
        """
        if train or np.random.rand() < epsilon:
            action = np.argmax(Q[s, :])
        else:
            action = np.random.randint(0, n_actions)
        return action

    def train_agent(self, alpha, gamma, epsilon, episodes, ep_step, max_steps, filename,render = False, test=False):
        """
        @param alpha learning rate
        @param gamma decay factor
        @param epsilon for exploration
        @param max_steps for max step in each episode
        """

        if(not path.exists(filename)):
            # Initialize q table
            n_actions = self.env.action_space.n
            n_states = self.env.observation_space.n
            q_table = np.zeros([n_states, n_actions])

            for episode in range(episodes):
                if episode % 100 == 0:
                    print(f"Episode: {episode}")

                state = self.env.reset()
                done = False

                # Explore or Exploit
                if np.random.rand() < epsilon:
                    action = np.argmax(q_table[state, :]) # Exploit learned values
                else:
                    action = np.random.randint(0, n_actions) # Explore action space
                t = 0
                while t < max_steps:

                    state_, reward, done, info = self.env.step(action)

                    if np.random.rand() < epsilon:
                        action_ = np.argmax(q_table[state_, :]) # Exploit learned values
                    else:
                        action_ = np.random.randint(0, n_actions) # Explore action space
                        
                    # Update values
                    old_value = q_table[state, action]       
                    if done:
                        new_value = old_value + alpha * (reward  - old_value)
                        q_table[state, action] = new_value
                        break
                    else:
                        new_value = old_value + alpha * (reward + (gamma * q_table[state_, action_] ) - old_value)
                        q_table[state, action] = new_value
                    state, action = state_, action_
                    t += 1

                if (ep_step != -1) and (episode % ep_step == 0):
                    self.Q = q_table
                    time, rew = self.simulate(filename = None,visualize=False, episodes=10)
                    self.stats['timesteps'].append(time)
                    self.stats['reward'].append(reward)
                    print("ep: {} -- saved: {}".format(ep_step,episode))

                # Decrease hyperparameters
                alpha -= self.alpha_decay
                gamma += self.gamma_decay
                epsilon += self.epsilon_decay

            # Save the values
            np.save(filename, q_table)

            print("Training finished.\n")


    def print_frames(self, frames):
        for i, frame in enumerate(frames):
            clear_output(wait=True)
            print(frame.get('frame'))
            print(f"Timestep: {i + 1}")
            print(f"State: {frame['state']}")
            print(f"Action: {frame['action']}")
            print(f"Reward: {frame['reward']}")
            sleep(.5)


    def simulate(self, filename, visualize, episodes):
        """Evaluate agent's performance after Q-learning"""
        total_epochs, total_penalties ,total_rewards = 0, 0, 0

        if(filename != None):
            q_table = np.load(filename)
        else:
            q_table = self.Q

        frames = [] # for animation

        for _ in range(episodes):
            state = self.env.reset()
            epochs, penalties, reward = 0, 0, 0
            
            done = False
            
            while not done:
                action = np.argmax(q_table[state])
                state, reward, done, info = self.env.step(action)

                total_rewards+=reward

                if reward == -10:
                    penalties += 1

                frames.append({
                    'frame': self.env.render(mode='ansi'),
                    'state': state,
                    'action': action,
                    'reward': reward
                    }
                )
                epochs += 1

            total_penalties += penalties
            total_epochs += epochs
            
        if visualize:  
            self.print_frames(frames)
        
        avg_timesteps = total_epochs / episodes
        avg_penalties = total_penalties / episodes
        avg_rewards = total_rewards / episodes

        return avg_timesteps, avg_rewards

    def getStats(self):
        return self.stats

    def resetStats(self):
        self.stats['timesteps'].clear()
        self.stats['reward'].clear()



