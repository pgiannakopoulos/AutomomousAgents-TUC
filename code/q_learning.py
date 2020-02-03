import gym
import numpy as np
import random
from IPython.display import clear_output
from time import sleep
import os.path
from os import path

class Agent_QL:
    def __init__(self, env, alpha_decay = 0.0001, gamma_decay = 0.0001, epsilon_decay = 0.0001):
        self.env = env

        self.Q = None
        self.alpha_decay = alpha_decay
        self.gamma_decay = gamma_decay
        self.epsilon_decay = epsilon_decay

        self.stats = dict()
        self.stats['timesteps'] = list()
        self.stats['reward'] = list()

    def print_frames(self, frames):
        for i, frame in enumerate(frames):
            clear_output(wait=True)
            print(frame.get('frame'))
            print(f"Timestep: {i + 1}")
            print(f"State: {frame['state']}")
            print(f"Action: {frame['action']}")
            print(f"Reward: {frame['reward']}")
            sleep(.5)

    def train_agent(self, alpha, gamma, epsilon, episodes, ep_step, max_steps, filename):
        """Training the agent"""

        if(not path.exists(filename)):
            # Initialize q table
            q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])

            for episode in range(episodes):
                if episode % 100 == 0:
                    print(f"Episode: {episode}")

                state = self.env.reset()
                done = False
                
                t = 0
                while t < max_steps:

                    # Explore or Exploit
                    if np.random.rand() < epsilon:
                        action = np.argmax(q_table[state]) # Exploit learned values
                    else:
                        action = self.env.action_space.sample() # Explore action space

                    next_state, reward, done, info = self.env.step(action) 
                    
                    old_value = q_table[state, action]
                    next_max = np.max(q_table[next_state])

                    # Update values
                    if done:
                        new_value = old_value + alpha * (reward - old_value)
                        q_table[state, action] = new_value
                        break
                    else:
                        new_value = old_value + alpha * (reward + (gamma * next_max) - old_value)
                        q_table[state, action] = new_value

                    state = next_state

                    t += 1

                if (ep_step != -1) and (episode % ep_step == 0):
                    self.Q = q_table
                    time, rew = self.simulate(filename = None,visualize=False, episodes=10)
                    self.stats['timesteps'].append(time)
                    self.stats['reward'].append(reward)
                 
                # Decrease hyperparameters
                if alpha - self.alpha_decay > 0:
                    alpha -= self.alpha_decay
                if  gamma + self.gamma_decay < 1:
                    gamma += self.gamma_decay
                if epsilon + self.epsilon_decay < 1:
                    epsilon += self.epsilon_decay      
            
            # Save the values
            np.save(filename, q_table)

            print("Training finished.\n") 

    def simulate(self,filename, visualize, episodes):
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


