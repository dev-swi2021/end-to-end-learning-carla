"""
   CIL 기반 신경망
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functionatl as F
import torch.optim as optim
from torchsummary import summary

from util import *

class RandomProcess(object):
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma
    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class OrnsteinUhlenbeckNoise(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0, sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckNoise, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_state()
    
    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * n_steps_annealing)
        self.x_prev = x
        self.n_steps += 1

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)
    

class Memory:
    def __init__(self, capacity):
        self.memory = deque(capacity)
        self.cnt = 0
        self.capacity = capacity
            
    def add(self, *data):
        self.memory.append(*data)
        self.cnt = (self.cnt+1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class Network(nn.Module):
    """
    Perception - + BarchNormalization Layer, ReLU Activation Layer
       Input Dim   output Ch  #Kernels   Stride     Dropout
    200 x 88 x   M    32        5          2          0.0
     98 x 48 x  32    32        3          1          0.0  
     96 x 46 x  32    64        3          2          0.0
     47 x 22 x  64    64        3          1          0.0
     45 x 20 x  64   128        3          2          0.0
     22 x  9 x 128   128        3          1          0.0
     20 x  7 x 128   256        3          2          0.0
      9 x  3 x 256   256        3          1          0.0
      7 x  1 x 256   512        -          -          0.0
          512        512        -          -          0.0

    Measurement
       Input Dim   output Ch  #Kernels   Stride     Dropout
           1         128        -          -          0.0
         128         128        -          -          0.0

    Join
       Input Dim   output Ch  #Kernels   Stride     Dropout
        512+128      512        -          -          0.3
    
    Action Branch
       Input Dim   output Ch  #Kernels   Stride     Dropout
          512        256        -          -          0.5
          256        256        -          -          0.5
          256          3        -          -          0.0
    
    Speed Branch
       Input Dim   output Ch  #Kernels   Stride     Dropout
          512        256        -          -          0.5
          256        256        -          -          0.5
          256          1        -          -          0.0

    """
    def __init__(self, channels):
        self.perception = self.build_network(channels)
    
    def build_network(self, channels):
        return nn.Sequential()
    
    def forward(self, *x):
        return self.network(x)


class ConditionalIL(nn.Module):
    def __init__(self, channels):
        self.network = Network(channels)
    
    def train(self):
        pass

    def eval(self):
        pass    
    
### Test 용###
class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w,init_w)
        

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.tanh(self.fc3(out))
        return out


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu - nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w,init_w)
    
    def forward(self, xs):
        x,a = xs
        out = self.relu(self.fc1(x))
        out = self.fc2(torch.cat([out, a],1))
        out = self.relu(out)
        out = self.fc3(out)
        return out


class DDPG(object):
    def __init__(self, nb_states, nb_actions, args):
        """
            network, noise, memory
        """
        if args.seed > 0:
            self.seed(args.seed)
        self.nb_states, self.nb_actions = nb_states, nb_actions
        self.capacity = args.capacity
        net_cfg = {
            'hidden1' : args.hidden1,
            'hidden2' : args.hidden2,
            'init_w' : args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.target_actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.target_critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.target_optim = optim.Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
         
        self.memory = Memory(args.capacity)
        self.noise = OrnsteinUhlenbeckNoise(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        self.epsilon = 1.0
        self.s_t = None
        self.a_t = None
        self.is_training = True

        if USE_CUDA: self.cuda()

    def update_policy(self):
        state_batch, action_batch, reward_batch, next_state_batch, termonal_batch = self.memory.sample(self.batch_size)

        next_q_values = self.target_critic([to_tensor(next_state_batch, volatile=True), self.target_actor(to_tensor(next_state_batch, volatile=True))])
        next_q_values.volatile = False

        target_q_batch = to_tensor(reward_batch) + self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic Update
        self.critic.zero_grad()
        q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor Update
        self.actor.zero_grad()
        policy_loss = -self.critic([to_tensor(state_batch), self.actor(to_tensor(state_batch))])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)
    
    def eval(self):
        self.actor.eval()
        self.target_actor.eval()
        self.critic.eval()
        self.target_critic.eval()
    
    def cuda(self):
        self.actor.cuda()
        self.target_actor.cuda()
        self.critic.cuda()
        self.target_critic.cuda()
    
    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1
    
    def random_action(self):
        action = np.random.uniform(-1., 1., self.nb_actions)
        self.a_t = action
        return action
    
    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(self.actor(to_tensor(np.array([s_t]))).squeeze(0)
        action += self.is_training*max(self.epsion, 0)*self.noise.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action
    
    def reset(self, obs):
        self.s_t = obs
        self.noise.reset_states()
    
    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )
    
    def save_model(self, output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )
    
    def seed(self, s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)