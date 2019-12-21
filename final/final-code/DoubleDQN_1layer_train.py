import gym
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque


class double_DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=10000)
        self.gamma = 0.9
        self.tau = 0.5
        self.epsilon = 0.1
        self.learning_rate = 0.001
        self.batch_size = 50
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(self.env.action_space.n, input_dim=state_shape[0]))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        samples = random.sample(self.memory, self.batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                action_max = np.argmax(self.model.predict(new_state)[0])
                Q_future = self.target_model.predict(new_state)[0][action_max]
                target[0][action] = reward + self.gamma * Q_future
            self.model.fit(state, target, epochs=1, verbose=0)

    def reset_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


def main():
    env = gym.make("Acrobot-v1")
    episodes = 3000
    env._max_episode_steps = 2000
    epi_len = 2000
    update_len = 100
    reset_len = 2000
    total_step = 0
    reward_total_chain = np.zeros(episodes)
    ddqn_agent = double_DQN(env=env)

    start_time = datetime.datetime.now()

    for i_episode in range(episodes):
        cur_state = env.reset().reshape(1, 6)
        reward_total = 0.0
        for step in range(epi_len):
            # env.render()
            action = ddqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape(1, 6)
            ddqn_agent.remember(cur_state, action, reward, new_state, done)
            total_step += 1
            if (total_step % update_len == 0) & (total_step > 0):
                ddqn_agent.train()
            if (total_step % reset_len == 0) & (total_step > 0):
                ddqn_agent.reset_target()
            cur_state = new_state
            reward_total += reward
            if done:
                break

        reward_total_chain[i_episode] = reward_total
        print("For trial {}".format(i_episode),  ", step = ", step)

        if i_episode >= 200:
            ave_reward_total = np.mean(reward_total_chain[range(i_episode - 50, i_episode)])
            if ave_reward_total > -100:
                break

    end_time = datetime.datetime.now()
    interval = (end_time - start_time).seconds
    final_time = interval / 60.0
    print('final_time:\t', final_time)

    ddqn_agent.save_model("model_1layer_1")
    np.save("reward_total_chain_1layer_1.npy", reward_total_chain)


if __name__ == "__main__":
    main()


# 6.1 min


result = np.load('reward_total_chain_1layer_1.npy')

plt.figure()
plt.plot(result[range(328)])
plt.xlabel('Episodes')
plt.ylabel('Total Rewards')
plt.savefig("train_reward_1layer_1.png")

