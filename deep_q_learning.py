import numpy as np
import gym
import _pickle as cPickle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Embedding, Reshape
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import tensorflow as tf
import os.path
from os import path

tf.compat.v1.disable_eager_execution()

#  pip3 install tensorflow==2.0.0-beta

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class DQN_Agent:
	def __init__(self, env):
		self.env = env
		np.random.seed(1)
		self.nb_actions = env.action_space.n
		self.nb_observations = env.observation_space.n
		model = Sequential()
		model.add(Embedding(self.nb_observations,10,input_length=1))
		model.add(Reshape((10,)))
		model.add(Dense(50, activation='relu'))
		model.add(Dense(50, activation='relu'))
		model.add(Dense(50, activation='relu'))
		model.add(Dense(self.nb_actions, activation='linear'))
		print(model.summary())
		memory = SequentialMemory(limit=50000, window_length=1)
		policy = EpsGreedyQPolicy()
		self.dqn = DQNAgent(model=model, nb_actions=self.nb_actions, memory=memory, nb_steps_warmup=500,target_model_update=1e-2, policy=policy)
		self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])

	def train_agent(self, filename):	
		if(not path.exists(filename)):
			self.dqn.fit(env, nb_steps=230000, visualize=False, verbose=1, nb_max_episode_steps=99, log_interval=2000)
			# After training is done, we save the final weights.
			self.dqn.save_weights(filename, overwrite=True)
		else:
			self.dqn.load_weights(filename)

	def simulate(self, visualze, episodes):
		visual = visualze
		stats = self.dqn.test(self.env, nb_episodes=episodes, visualize=visual, nb_max_episode_steps=99)
		return np.mean(stats.history['episode_reward']),np.mean(stats.history['nb_steps'])
