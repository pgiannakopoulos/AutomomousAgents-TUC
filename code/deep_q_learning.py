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
from time import sleep

tf.compat.v1.disable_eager_execution()

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

class DQN_Agent:
	def __init__(self, env):
		self.env = env

		self.stats = dict()
		self.stats['timesteps'] = list()
		self.stats['reward'] = list()

		np.random.seed(1)
		self.nb_actions = env.action_space.n
		self.nb_observations = env.observation_space.n

		# Create the model
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

	# Train the agent
	def train_agent(self, episodes, ep_step, filename):	
		if(not path.exists(filename)):
			if ep_step == -1:
				self.dqn.fit(self.env, nb_steps=episodes, visualize=False, verbose=1, nb_max_episode_steps=99, log_interval=2000)
			else:
				for episode in range(ep_step,episodes+1, ep_step):
					data = self.dqn.fit(self.env, nb_steps=episode, visualize=False, verbose=1, nb_max_episode_steps=99, log_interval=2000)
					time, rew = self.simulate(visualize = False, episodes = 10)
					self.stats['timesteps'].append(time)
					self.stats['reward'].append(rew)
			self.dqn.save_weights(filename, overwrite=True)
		else:
			self.dqn.load_weights(filename)
		
	# Evaluate the agent
	def simulate(self, filename, visualize, episodes):
		if(not path.exists(filename)):
			train_agent(episodes = 2000, ep_step=-1, filename=filename)
		else:
		 	self.dqn.load_weights(filename) 
		visual = visualize
		data = self.dqn.test(self.env, nb_episodes=episodes, visualize=visual, nb_max_episode_steps=99)
		return np.mean(data.history['nb_steps']),np.mean(data.history['episode_reward'])

	def getStats(self):
		return self.stats

	def resetStats(self):
		self.stats['timesteps'].clear()
		self.stats['reward'].clear()

	def loadWeights(self, filename):
		self.dqn.load_weights(filename)