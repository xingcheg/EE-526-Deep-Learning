import gym
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('model_2layers_1')

env = gym.make('Acrobot-v1')
env._max_episode_steps = 500
reward_total_chain = np.zeros(500)


for i_episode in range(500):
    state = env.reset().reshape(1, 6)
    reward_total = 0
    for t in range(1000):
        action = np.argmax(model.predict(state))
        state, reward, done, info = env.step(action)
        state = state.reshape(1, 6)
        reward_total += reward
        if done:
            break
    reward_total_chain[i_episode] = reward_total


np.mean(reward_total_chain)
np.median(reward_total_chain)
np.std(reward_total_chain)


plt.hist(reward_total_chain, bins='auto')
plt.xlabel('Total Rewards')
plt.ylabel('Frequency')
plt.savefig('histogram_m2.png')


