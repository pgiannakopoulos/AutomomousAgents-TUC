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
    def __init__(self, env):
        self.env = env

    def init_q(s, a, type="ones"):
        """
        @param s the number of states
        @param a the number of actions
        @param type random, ones or zeros for the initialization
        """
        if type == "ones":
            return np.ones((s, a))
        elif type == "random":
            return np.random.random((s, a))
        elif type == "zeros":
            return np.zeros((s, a))


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

    def train_agent(self,alpha, gamma, epsilon, episodes, max_steps, n_tests, filename,render = False, test=False):
        """
        @param alpha learning rate
        @param gamma decay factor
        @param epsilon for exploration
        @param max_steps for max step in each episode
        @param n_tests number of test episodes
        """
        timestep_reward = []
        if(not path.exists(filename)):
            n_states, n_actions = self.env.observation_space.n, self.env.action_space.n
            Q = self.init_q(n_states, n_actions, type="ones")
            for episode in range(episodes):
                print(f"Episode: {episode}")
                total_reward = 0
                s = self.env.reset()
                a = epsilon_greedy(Q, epsilon, n_actions, s)
                t = 0
                done = False
                while t < max_steps:
                    if render:
                        self.env.render()
                    t += 1
                    s_, reward, done, info = self.env.step(a)
                    total_reward += reward
                    a_ = epsilon_greedy(Q, epsilon, n_actions, s_)
                    if done:
                        Q[s, a] += alpha * ( reward  - Q[s, a] )
                    else:
                        Q[s, a] += alpha * ( reward + (gamma * Q[s_, a_] ) - Q[s, a] )
                    s, a = s_, a_
                    if done:
                        if render:
                            print(f"This episode took {t} timesteps and reward {total_reward}")
                        timestep_reward.append(total_reward)
                        break
            if render:
                print(f"Here are the Q values:\n{Q}\nTesting now:")
            if test:
                test_agent(Q, self.env, n_tests, n_actions)

            np.save(filename, Q)

            print("Training finished.\n")
        return timestep_reward

    def test_agent(Q, env, n_tests, n_actions, delay=0.1):
        for test in range(n_tests):
            print(f"Test #{test}")
            s = env.reset()
            done = False
            epsilon = 0
            total_reward = 0
            while True:
                time.sleep(delay)
                env.render()
                a = epsilon_greedy(Q, epsilon, n_actions, s, train=True)
                print(f"Chose action {a} for state {s}")
                s, reward, done, info = env.step(a)
                total_reward += reward
                if done:
                    print(f"Episode reward: {total_reward}")
                    time.sleep(1)
                    break

    def print_frames(self, frames):
        for i, frame in enumerate(frames):
            clear_output(wait=True)
            print(frame.get('frame'))
            print(f"Timestep: {i + 1}")
            print(f"State: {frame['state']}")
            print(f"Action: {frame['action']}")
            print(f"Reward: {frame['reward']}")
            sleep(.5)


    def simulate(self):
        """Evaluate agent's performance after Q-learning"""
        total_epochs, total_penalties = 0, 0
        episodes = 1
        Q = np.load('training_sarsa.npy')
        frames = [] # for animation

        for _ in range(episodes):
            state = self.env.reset()
            epochs, penalties, reward = 0, 0, 0
            
            done = False
            
            while not done:
                action = np.argmax(Q[state])
                state, reward, done, info = self.env.step(action)

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
            
        self.print_frames(frames)
        print(f"Results after {episodes} episodes:")
        print(f"Average timesteps per episode: {total_epochs / episodes}")
        print(f"Average penalties per episode: {total_penalties / episodes}")


