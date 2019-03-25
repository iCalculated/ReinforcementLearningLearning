import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


#from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v0"
GAMMA = 0.95
TAU = 0.001
LEARNING_RATE = 0.001

MEMORY_SIZE = 100000
BATCH_SIZE = 20

MAX_EPSILON = 1.0
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.9995


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.epsilon = MAX_EPSILON

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        #24/24 start
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        #self.model.add(Dense(5,activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, done in batch:
            q_update = reward
            if not done:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.epsilon *= EPSILON_DECAY
        self.epsilon = max(MIN_EPSILON, self.epsilon)


def cartpole():
    env = gym.make(ENV_NAME)
    env.seed(1024)
    #score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            if run > 1:
                env.render()
            action = dqn_solver.act(state)
            state_next, reward, done, info = env.step(action)
            reward = reward if not done else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, done)
            state = state_next

            if done:
                #if step != 200:
                reward += state[0][0]
                print("Step: " + str(step) + ", Run: " + str(run) + ", exploration: " + str(dqn_solver.epsilon) + ", score: " + str(reward))
                #core_logger.add_score(step, run)
                break
            dqn_solver.experience_replay()


if __name__ == "__main__":
    cartpole()