import numpy as numpy
import matplotlib.pyplot as plt
from scipy.io import savemat

from util import *

class Evaluator(object):
    def __init__(self, num_episodes, interval, save_path='', max_episode_length=None):
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.save_path = save_path
        self.results = np.array([]).reshape(num_episodes,0)
    
    def __call__(self, env, policy, debug=False, visualize=False, save=True):
        self.is_training = False
        observation = None
        result = []

        for episode in range(self.num_episodes):
            observation = env.reset()
            episode_steps = 0
            episode_rward = 0.

            assert observation is not None

            done = False
            while not done:
                action = policy(observation)

                observation, reward, done, _ = env.step(action)
                if self.max_episode_length and episode_steps >= self.max_episode_length - 1:
                    done = True
                if visualize:
                    env.render(mode='human')
                
                episode_reward += reward
                episode_steps += 1
            
            result.append(episode_reward)
        result = np.array(result).reshape(-1, 1)
        self.results = np.hstack([self.results, result])

        if save:
            self.save_results('{}/validate_reward'.format(self.save_path))
        return np.mean(result)
    
    def save_results(self, fn):
        y = np.mean(self.results, axis=0)
        error = np.std(self.results, axis=0)

        x = range(0, self.results.shape[1] * self.interval, self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.save_fig(fn+'.png')
        savemat(fn+'.mat', {'reward':self.results})