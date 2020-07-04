import argparse

import gym
from gym import wrappers, logger

# my imports
from nn import *
import collections
import numpy as np
import random
import pandas as pd
from PIL import Image, ImageDraw





# TODO Die gewünschte Anzahl an zu spielenden Spielen
episode_count = 100

# TODO Verzeichnis für die Ergebnisse der Spiele
# Default unter gym/results
outdir = './results'



class PacmanAgent(object):

    def __init__(self, env):
        # env variables
        self.env = env
        self.actions = self.env.action_space.n
        self.buffer_frames = 4
        self.img_shape = (84, 84, 3*self.buffer_frames) # pixels, pixels, rgb
        self.num_values = 1
        self.gamma = 0.95
        self.lr_actor = 0.0003
        self.lr_critic = 0.0003
        self.model = NN(self.img_shape, self.actions, self.num_values, self.lr_actor, self.lr_critic)
        
        self.best_mean_reward = 0.0


    def act(self, state):
        
        policy = self.model.predict_actor(state)[0]
        action_int = np.random.choice(self.actions, p=policy)
        
              
        
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

    def update_policy(self, state, action_int, reward, next_state, done):
        values = np.zeros((1, self.num_values))
        advantages = np.zeros((1, self.actions))

        value = self.model.predict_critic(state)[0]
        next_value = self.model.predict_critic(next_state)[0]

        if done:
            advantages[0][action_int] = reward - value
            values[0][0] = reward
        else:
            advantages[0][action_int] = (reward + self.gamma * next_value) - value
            values[0][0] = reward + self.gamma * next_value

        #print("Values:", values, "Advantages:", advantages)

        self.model.train_actor(state, advantages)
        self.model.train_critic(state, values)


    def train(self, num_episodes):
        total_rewards = []

        for episode in range(num_episodes):
            total_reward = 0.0
            state = self.env.reset()

            while True:
                action, action_int = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                if done and total_reward < 85:
                    reward = -100

                self.update_policy(state, action_int, reward, next_state, done)

                total_reward += reward
                state = next_state

                if done:
                    if total_reward < 85:
                        total_reward += 100
                    total_rewards.append(total_reward)
                    mean_reward = np.mean(total_rewards[-5:])
                    

                    print("Episode:", episode+1, "\tReward:", total_reward, "\tMean:", mean_reward)

                    
                    # Saving
                    if episode+1 == 100:
                        self.model.save_actor("pacman_100runs.h5")
                    elif episode+1 == 500:
                        self.model.save_actor("pacman_500runs.h5")
                    elif episode+1 == 1000:
                        self.model.save_actor("pacman_1000runs.h5")
                    elif episode+1 == 2500:
                        self.model.save_actor("pacman_2500runs.h5")
                    elif episode+1 == 5000:
                        self.model.save_actor("pacman_5000runs.h5")

                    if mean_reward > 85:
                        self.model.save_actor("pacman_good_rewards.h5")
                    if mean_reward > self.best_mean_reward:
                        self.model.save_actor("pacman_best_rewards.h5")
                        self.best_mean_reward = mean_reward
                    break
        return total_rewards

    
    def play(self, num_episodes):
        self.model.load_model('pacman_best_rewards.h5')

        for episode in range(num_episodes):
            state = self.env.reset()

            while True:
                action, _ = self.act(state)
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
    total_rewards = agent.train(num_episodes=5000)
    df = pd.DataFrame(total_rewards, columns=['Rewards'])
    df.to_csv("total_rewards.csv", index=False, mode="a", header="False")
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
    

