import gym
import numpy as np
import random
from IPython.display import clear_output
from time import sleep
import os.path
from os import path

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame.get('frame'))
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.5)

def train(env):
    """Training the agent"""
    # state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
    # env.P[328] {action: [(probability, nextstate, reward, done)]}
    action_man = {0 : "south", 1 : "north", 2 : "east", 3 : "west", 4 : "pickup", 5 : "dropoff"}

    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # Hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    # For plotting metrics
    all_epochs = []
    all_penalties = []

    for i in range(1, 100001):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            next_state, reward, done, info = env.step(action) 
            
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

    np.save('training_ql.npy', q_table)

    print("Training finished.\n")

def simulate(env):
    """Evaluate agent's performance after Q-learning"""
    total_epochs, total_penalties = 0, 0
    episodes = 1
    q_table = np.load('training_ql.npy')
    frames = [] # for animation

    for _ in range(episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
        
        done = False
        
        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)

            if reward == -10:
                penalties += 1

            frames.append({
                'frame': env.render(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward
                }
            )
            epochs += 1

        total_penalties += penalties
        total_epochs += epochs
        
    print_frames(frames)
    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")


env = gym.make("Taxi-v3")

if(not path.exists('training_ql.npy')):
    train(env)

simulate(env)