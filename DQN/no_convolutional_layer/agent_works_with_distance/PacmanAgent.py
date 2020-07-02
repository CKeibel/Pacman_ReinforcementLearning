import argparse

import gym
from gym import wrappers, logger

# my imports
from dqn import *
import collections
import numpy as np
import random




# TODO Die gewünschte Anzahl an zu spielenden Spielen
episode_count = 100

# TODO Verzeichnis für die Ergebnisse der Spiele
# Default unter gym/results
outdir = './results'



class PacmanAgent(object):

    def __init__(self, env):
        # env variables
        self.env = env
        self.observations = self.env.observation_space.shape
        self.actions = self.env.action_space.n

        # DQN Agent Variables
        self.replay_buffer_size = 60000
        self.train_start = 3000
        self.memory = collections.deque(maxlen=self.replay_buffer_size)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        
        # DQN Network Variables
        self.state_shape = self.observations
        self.lr = 1e-3
        self.model = DQN(self.state_shape, self.actions, self.lr)
        self.target_model = DQN(self.state_shape, self.actions, self.lr)
        self.target_model.update_model(self.model)
        self.batch_size = 64

        #self.action_space = action_space

    def act(self, state):
        # Epsilon Greedy
        action_int = 0
        if np.random.rand() <= self.epsilon:
            action_int = self.env.action_space.sample() # Exploration
        else:
            action_int = np.argmax(self.model.predict(state)) # Exploitation

        #action_int = np.random.randint(self.actions)
              
        
        if action_int == 0:
            action = "GO_NORTH"
        elif action_int == 1:
            action = "GO_WEST"
        elif action_int == 2:
            action = "GO_EAST"
        elif action_int == 3:
            action = "GO_SOUTH"
        elif action_int == 4:
            action = "QUIT_GAME"
        elif action_int == 5:
            action = "WAIT"
        
        return action, action_int

    def train(self, num_episodes):
        total_rewards = []

        for episode in range(num_episodes):
            state = self.env.reset()
            state = np.reshape(state, (1, state.shape[0]))
            total_reward = 0.0
            steps = 0
            

            while True:
                action, action_int = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, (1, next_state.shape[0]))
                steps += 1
                #print(state)
                #print("Steps", steps, "\tRewards:", reward, "\tTotal:", total_reward)
                #print(reward, total_reward)
                #print(reward)
                if done and total_reward < 7:
                    reward = -100

                self.remember(state, action_int, reward, next_state, done)
                self.replay()
                total_reward += reward
                state = next_state
                

                if done:
                    if total_reward < 7:
                        total_reward += 100
                    total_rewards.append(total_reward)
                    mean_reward = np.mean(total_rewards[-5:])
                    if mean_reward > 6:
                        self.model.save_model("pacman_good_reward.h5")
                        return
                    self.target_model.update_model(self.model)
                    print("Episode:", episode+1, "Total Reward:", total_reward, "Mean:", mean_reward, "Steps:", steps, "EPSI:", self.epsilon)
                    break

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, states_next, dones = zip(*minibatch)

        states = np.concatenate(states)
        states_next = np.concatenate(states_next)

        q_values = self.model.predict(states)
        q_values_next = self.target_model.predict(states_next)

        for i in range(self.batch_size):
            a = actions[i]
            done = dones[i]
            if done:
                q_values[i][a] = rewards[i]
            else:
                q_values[i][a] = rewards[i] + self.gamma * np.max(q_values_next[i])

        self.model.train(states, q_values)

    def play(self, num_episodes):
        self.model.load_model("Pacman_1000_runs.h5")

        for episode in range(num_episodes):
            state = self.env.reset()
            state = np.reshape(state, (1, state.shape[0]))
            total_reward = 0.0

            while True:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, (1, next_shape.shape[0]))
                state = next_state
                total_reward += reward
                
                if done:
                    print("Reward:", total_reward)
                    break



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='gym_pacman_environment:test-v0', help='Select the environment to run')
    args = parser.parse_args()

    # Hier könnt ihr das Level auf logger.DEBUG or logger.WARN setzen, um mehr Details auszugeben
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    if outdir is None:
        raise Exception("Please set the directory where to put the results into")
    
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = PacmanAgent(env)
    agent.train(num_episodes=1000)
    agent.model.save_model('Pacman_1000_runs.h5')
    
       

    # for i in range(1):
    #     state = env.reset()
    #     total_reward = 0
    #     while True:
    #         action = agent.act(state)
    #         state, reward, done, debug = env.step(action)
    #         #print(state)
    #         print(reward)
    #         total_reward += reward
    #         if done:
    #             break
    #     print(total_reward)
    
    env.close()
    

