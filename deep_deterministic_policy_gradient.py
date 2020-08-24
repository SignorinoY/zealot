from reinforcement_learning.deep_deterministic_policy_gradient import DDPG
from reinforcement_learning.utils.memory import Memory, Transition
from reinforcement_learning.env.arm import Arm

import torch
import random
import os

EPISODES = 10000
STEPS = 200
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = Arm()

s_dim = env.observation_dim
a_dim = env.action_dim
a_bound = env.action_space

model = DDPG(a_dim, s_dim, a_bound, 0.9, 0.01)
memory = Memory(30000)

def train():
    if os.path.isfile('./arm-ddpg'):
        model.load('./arm-ddpg')
    for i in range(EPISODES):
        score = 0
        value_loss, policy_loss = 0, 0
        state = torch.Tensor([env.reset()]).to(device)
        for j in range(STEPS):
            # env.render()
            action = model.predict(state)
            next_state, reward, done = env.step(action.cpu().numpy()[0])
            score += reward
            mask = torch.Tensor([done]).to(device)
            reward = torch.Tensor([reward]).to(device)
            next_state = torch.Tensor([next_state]).to(device)
            memory.push(state, action, mask, next_state, reward)
            state = next_state
            if len(memory) > BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                value_loss, policy_loss = model.train(batch)
            if done:
                break
        if done:
            model.save('./arm-ddpg')
        print('Episode %d:%s Critic Loss %.2f, Action Loss %.2f' % (i+1, ' Finished Step %d,' % (j+1) if done else '', value_loss, policy_loss))


def test():
    model.load('./arm-ddpg')
    for _ in range(10):
        state = torch.Tensor([env.reset()]).to(device)
        while True:
            env.render()
            action = model.predict(state)
            next_state, reward, done = env.step(action.cpu().numpy()[0])
            state = torch.Tensor([next_state]).to(device)
            if done:
                break

train()
