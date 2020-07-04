import argparse

import gym
from gym import wrappers, logger

# my imports
from dqn import *
import collections
import numpy as np
import random
import pandas as pd
from PIL import Image, ImageDraw





# TODO Verzeichnis für die Ergebnisse der Spiele
# Default unter gym/results
outdir = './results'



class PacmanAgent(object):

    def __init__(self, env):
        # env variables
        self.env = env
        self.actions = self.env.action_space.n
        self.img_shape = (84, 84, 3) # pixels, pixels, rgb

        # DQN Agent Variables
        self.replay_buffer_size = 100000
        self.train_start = 1000
        self.memory = collections.deque(maxlen=self.replay_buffer_size)
        self.gamma = 0.995
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_steps = 100000
        self.epsilon_step = (self.epsilon - self.epsilon_min) / self.epsilon_steps        
        # DQN Network Variables
        self.lr = 0.0003
        self.model = DQN(self.img_shape, self.actions, self.lr)
        self.target_model = DQN(self.img_shape, self.actions, self.lr)
        self.target_model.update_model(self.model)
        self.batch_size = 32
        self.sync_models = 100
        self.path_model = "pacman_model.h5"
        self.path_target_model = "pacman_targetmodel.h5"
        self.load = False
        if self.load:
            self.model.load_model(self.path_model)
            self.target_model.load_model(self.path_target_model)

        self.best_mean_reward = 0.0


    def act(self, state, flag):
        
        action_int = 0
        
        # flag unterscheidet ob im Trainingsmodus(TRUE) oder Playmodus(FALSE)
        if flag:
            # Epsilon Greedy
            if np.random.rand() <= self.epsilon:
                action_int = self.env.action_space.sample() # Exploration
            else:
                action_int = np.argmax(self.model.predict(state)) # Exploitation
        else:
            action_int = np.argmax(self.model.predict(state))
              
        
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
        it = 0

        for episode in range(num_episodes):
            total_reward = 0.0
            state = self.env.reset()

            while True:
                it += 1
                action, action_int = self.act(state, True)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done and total_reward < 60:
                    reward = -100

                self.remember(state, action_int, reward, next_state, done)
                self.epsilon_anneal()
                self.replay()
                state = next_state

                if it % self.sync_models == 0:
                    self.target_model.update_model(self.model)

                if done:
                    if total_reward < 60:
                        total_reward += 100 # only for output
                    total_rewards.append(total_reward)
                    mean_reward = np.mean(total_rewards[-5:])
                    df = pd.DataFrame([total_reward], columns=['Rewards'])
                    df.to_csv("total_rewards.csv", index=False, mode="a", header="False")

                    print("Episode:", episode+1, "\tMemSize:", len(self.memory), "\tReward:", total_reward, "\tMean:", mean_reward, "\tE:", self.epsilon)
                    self.model.save_model(self.path_model)
                    self.target_model.save_model(self.path_target_model)
                    
                    # Saving
                    if episode+1 == 100:
                        self.model.save_model("pacman_100runs.h5")
                    elif episode+1 == 500:
                        self.model.save_model("pacman_500runs.h5")
                    elif episode+1 == 1000:
                        self.model.save_model("pacman_1000runs.h5")
                    elif episode+1 == 2500:
                        self.model.save_model("pacman_2500runs.h5")
                    elif episode+1 == 5000:
                        self.model.save_model("pacman_5000runs.h5")

                    if mean_reward > 55:
                        self.model.save_model("pacman_good_rewards.h5")
                    if mean_reward > self.best_mean_reward:
                        self.model.save_model("pacman_best_rewards.h5")
                        self.best_mean_reward = mean_reward
                    break
            
    def epsilon_anneal(self):
        if len(self.memory) < self.train_start:
            return
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_step

    def remember(self, state, action_int, reward, next_state, done):
        self.memory.append([state, action_int, reward, next_state, done])

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
        self.model.load_model('pacman_best_rewards.h5')
        self.target_model.load_model(self.path_target_model)

        for episode in range(num_episodes):
            state = self.env.reset()

            while True:
                action, _ = self.act(state, False)
                state, _, done, _ = self.env.step(action)                
                if done:
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
    agent.train(num_episodes=5000)
    #agent.play(2)
    
       

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
    

