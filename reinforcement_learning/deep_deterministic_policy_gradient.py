import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import gc

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class Actor(nn.Module):

    def __init__(self, a_dim, s_dim, a_bound):
        
        super(Actor, self).__init__()
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]

        # Hidden Layer 1
        self.fc1 = nn.Linear(s_dim, 300)
        # Hidden Layer 2
        self.fc2 = nn.Linear(300, 300)
        # Output Layer
        self.mu = nn.Linear(300, a_dim)

    def forward(self, s):

        x = s
        # Hidden Layer 1
        x = F.relu(self.fc1(x))
        # Hidden Layer 1
        x = F.relu(self.fc2(x))
        # Output Layer
        mu = torch.tanh(self.mu(x)) * self.a_bound
        return mu

class Critic(nn.Module):

    def __init__(self, a_dim, s_dim):
        
        super(Critic, self).__init__()
        self.a_dim, self.s_dim = a_dim, s_dim

        # Hidden Layer 1
        self.fc1 = nn.Linear(s_dim + a_dim, 300)
        # Hidden Layer 2
        self.fc2 = nn.Linear(300, 300)
        # Output Layer
        self.V = nn.Linear(300, 1)

    def forward(self, a, s):

        x = torch.cat([a, s], 1)
        # Hidden Layer 1
        x = F.relu(self.fc1(x))
        # Hidden Layer 1
        x = F.relu(self.fc2(x))
        # Output Layer
        V = self.V(x)
        return V

class DDPG(object):
    
    def __init__(self, a_dim, s_dim, a_bound, gamma, tau):

        self.gamma = gamma
        self.tau = tau

        # Define the Actor Model
        self.actor = Actor(a_dim, s_dim, a_bound).to(device)
        self.actor_target = Actor(a_dim, s_dim, a_bound).to(device)

        # Defin the Critic Model
        self.critic = Critic(a_dim, s_dim).to(device)
        self.critic_target = Critic(a_dim, s_dim).to(device)

        # Define the Optimizer
        self.actor_optmizer = Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optmizer = Adam(self.critic.parameters(), lr=1e-3)

        # Make Actor, Critic Eval Model's parameters the same as Actor, Critic Model
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def predict(self, state):
        
        x = state.to(device)
        self.actor.eval()
        action = self.actor(x)
        self.actor.train()
        return action.data

    def train(self, batch):

        state = torch.cat(batch.state).to(device)
        action = torch.cat(batch.action).to(device)
        reward = torch.cat(batch.reward).to(device).unsqueeze(1)
        done = torch.cat(batch.done).to(device).unsqueeze(1)
        next_state = torch.cat(batch.next_state).to(device)

        target_Q = self.critic_target(self.actor_target(next_state), next_state)
        target_Q = reward + (done * self.gamma * target_Q).detach()

        current_Q = self.critic(action, state)

        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_optmizer.zero_grad()
        critic_loss.backward()
        self.critic_optmizer.step()

        actor_loss = - self.critic(self.actor(state), state).mean()
        self.actor_optmizer.zero_grad()
        actor_loss.backward()
        self.actor_optmizer.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return critic_loss.item(), actor_loss.item()

    def save(self, filepath):
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict()
        }
        torch.save(checkpoint, filepath)
        gc.collect()

    def load(self, filepath):
        key = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(filepath, map_location=key)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        gc.collect()