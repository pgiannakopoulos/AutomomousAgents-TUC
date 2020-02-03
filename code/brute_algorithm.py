import gym

# Brute algorith without learning
def brute_agent(env, episodes):
    env.reset()
    total_epochs, total_penalties ,total_rewards = 0, 0, 0

    epochs = 0
    penalties, reward = 0, 0

    for _ in range(episodes):
        done = False
        env.reset()
        epochs, penalties, reward = 0, 0, 0
        while not done:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            total_rewards+=reward

            if reward == -10:
                penalties += 1

            epochs += 1

            total_penalties += penalties
            total_epochs += epochs
          
    avg_timesteps = total_epochs / episodes
    avg_penalties = total_penalties / episodes
    avg_rewards = total_rewards / episodes

    return avg_timesteps, avg_rewards