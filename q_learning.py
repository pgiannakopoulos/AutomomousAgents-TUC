import gym
import numpy as np
import random
from IPython.display import clear_output
from time import sleep
import os.path
from os import path


class Agent_QL:
    def __init__(self, env):
        self.env = env

    def print_frames(self, frames):
        for i, frame in enumerate(frames):
            clear_output(wait=True)
            print(frame.get('frame'))
            print(f"Timestep: {i + 1}")
            print(f"State: {frame['state']}")
            print(f"Action: {frame['action']}")
            print(f"Reward: {frame['reward']}")
            sleep(.5)

    def train_agent(self, alpha, gamma, epsilon, episodes, filename):
        """Training the agent"""
        # state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
        # env.P[328] {action: [(probability, nextstate, reward, done)]}
        action_man = {0 : "south", 1 : "north", 2 : "east", 3 : "west", 4 : "pickup", 5 : "dropoff"}

        if(not path.exists(filename)):

            q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])

            # For plotting metrics
            all_epochs = []
            all_penalties = []

            for i in range(1, episodes+1):
                state = self.env.reset()

                epochs, penalties, reward, = 0, 0, 0
                done = False
                
                while not done:
                    if random.uniform(0, 1) < epsilon:
                        action = self.env.action_space.sample() # Explore action space
                    else:
                        action = np.argmax(q_table[state]) # Exploit learned values

                    next_state, reward, done, info = self.env.step(action) 
                    
                    old_value = q_table[state, action]
                    next_max = np.max(q_table[next_state])
                    
                    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                    q_table[state, action] = new_value

                    if reward == -10:
                        penalties += 1

                    state = next_state
                    epochs += 1
                    
                if i % 100 == 0:
                    clear_output(wait=True)
                    print(f"Episode: {i}")

            np.save(filename, q_table)

            print("Training finished.\n")

    def simulate(self,filename, visualize, episodes):
        """Evaluate agent's performance after Q-learning"""
        total_epochs, total_penalties ,total_rewards = 0, 0, 0
        q_table = np.load(filename)
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


