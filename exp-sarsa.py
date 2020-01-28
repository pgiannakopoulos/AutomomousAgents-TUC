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
Using the simplest gym environment for brevity: https://gym.openai.com/envs/FrozenLake-v0/
"""

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

def expected_sarsa(env, alpha, gamma, epsilon, episodes, max_steps, n_tests, render = False, test=False):
    """
    @param alpha learning rate
    @param gamma decay factor
    @param epsilon for exploration
    @param max_steps for max step in each episode
    @param n_tests number of test episodes
    """
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = init_q(n_states, n_actions, type="ones")
    timestep_reward = []
    for episode in range(episodes):
        print(f"Episode: {episode}")
        total_reward = 0
        s = env.reset()
        t = 0
        done = False
        while t < max_steps:
            if render:
                env.render()
            t += 1
            a = epsilon_greedy(Q, epsilon, n_actions, s)
            s_, reward, done, info = env.step(a)
            total_reward += reward
            if done:
                Q[s, a] += alpha * ( reward  - Q[s, a] )
            else:
                expected_value = np.mean(Q[s_,:])
                # print(Q[s,:], sum(Q[s,:]), len(Q[s,:]), expected_value)
                Q[s, a] += alpha * (reward + (gamma * expected_value) - Q[s, a])
            s = s_
            if done:
                print(f"This episode took {t} timesteps and reward {total_reward}")
                timestep_reward.append(total_reward)
                break
    if render:
        print(f"Here are the Q values:\n{Q}\nTesting now:")
    if test:
        test_agent(Q, env, n_tests, n_actions)

    np.save('training_esarsa.npy', Q)

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

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame.get('frame'))
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.5)


def simulate(env):
    """Evaluate agent's performance after Q-learning"""
    total_epochs, total_penalties = 0, 0
    episodes = 1
    Q = np.load('training_esarsa.npy')
    frames = [] # for animation

    for _ in range(episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
        
        done = False
        
        while not done:
            action = np.argmax(Q[state])
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

if __name__ =="__main__":
    env = gym.make('Taxi-v3')
    alpha = 0.4
    gamma = 0.999
    epsilon = 0.9
    episodes = 3000
    max_steps = 2500
    n_tests = 20
    if(not path.exists('training_esarsa.npy')):
        timestep_reward = expected_sarsa(env, alpha, gamma, epsilon, episodes, max_steps, n_tests)
    simulate(env)
    # print(timestep_reward)

