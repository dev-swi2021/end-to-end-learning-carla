import gym
import torch
import argparse
import numpy as np
from util import *
from copy import deepcopy
from model import DDPG
from evaluator import Evaluator

class NormalizedEnv(gym.ActionWrapper):
    def _action(self,action):
        act_k = (self.action_space.high - self.action_space.low) / 2.
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k * action + act_b
    
    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k_inv * (action - act_b)

def train(num_iterations, agent, env, evaluate, validate_steps, output, max_episode_length=None, debug=False):
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None

    while step < num_iterations:
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)
        
        if step <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)
        
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True
        
        agent.observe(reward, observation2, done)
        if step > args.warmup:
            agent.update_policy()
        
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            validate_reward = evaluate(env, policy, debug=False, visualize=False)
            if debug:
                print('[Evaluate] Step_{:07d}: mean_reward: {}'.format(step, validate_reward))
        
        if step % int(num_iterations/3) == 0:
            agent.save_model(output)
        
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:
            if debug: print("#{}: episode_reward:{} steps:{}".format(episode, episode_reward,step))

            agent.memory.append(
                observation,
                agent.select_action(observation),
                0, False
            )

            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1

def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):
    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug: print('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(decription='Pytorch on Gym')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='Pendulum-v0', type=str, help='open-ai gym environment')
    parser.add_argument('hidden1', default=400, type=int, help='hidden num of first fully connected layer')
    parser.add_argument('hidden2', default=300, type=int, help='hidden num of second fully connected layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--capacity', default=6000000, type=int, help='memory size')
    paresr.add_argument('--window_length', default=1, type=int)
    parser.add_argument('--tau', default=0.001, type=float, help='noise theta')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    paresr.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    paresr.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--validate_steps', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float)
    parser.add_argument('--train_iter', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    
    args =  parser.parse_args()
    if args.resume = 'default':
        args.resume = 'output/{}-run0'.format(args.env)
    
    env = NormalizedEnv(gym.make(args.env))

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)
    
    nb_states = en.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    agent = DDPG(nb_states, nb_actions, args)
    evaluate = Evaluator(args.validate_episodes, args.validate_steps, args.output, max_episode_length=args.max_episode_length)

    if args.mode == 'train':
        train(args.train_iter, agent, env, evaluate, args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug)
    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume, visualize=True, debug=args.debug)
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))