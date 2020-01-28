import gym

def brute_agent(env):
    env.reset()

    epochs = 0
    penalties, reward = 0, 0

    frames = [] # for animation

    done = False

    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1
        
        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
            }
        )

        epochs += 1
        
        
    print("Timesteps taken: {}".format(epochs))
    print("Penalties incurred: {}".format(penalties))