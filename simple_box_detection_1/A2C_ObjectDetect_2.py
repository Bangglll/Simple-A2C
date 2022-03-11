from simple_detection import Env

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
STATE_DIM = 4
ACTION_DIM = 2
VALUE = 1
hidden_size = 40


def log(logstr):
    file_name = os.path.join('./save_model2/', '220127_final2.log')
    with open(file_name, 'a') as f:
        print(logstr)
        f.write(logstr)
        f.write('\n')
        f.flush()

def plot(frame_idx, rewards):
    # clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. rewards: %s' % (frame_idx, rewards))
    plt.plot(rewards)
    # plt.show()
    plt.savefig("A2C_Obj_rewards_fig.png")

class A2C(nn.Module):
    def __init__ (self, action_size, state_size):
        super(A2C, self).__init__()
        self.actor_fc1 = nn.Linear(state_size, hidden_size)
        # self.actor_fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_fc3 = nn.Linear(hidden_size, action_size)

        self.critic_fc1 = nn.Linear(state_size, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, hidden_size)
        self.critic_fc3 = nn.Linear(hidden_size,1)

    def forward(self, x):
        actor_x = F.tanh(self.actor_fc1(x))
        # actor_x = F.tanh(self.actor_fc2(actor_x))
        policy =  F.softmax(self.actor_fc3(actor_x))

        critic_x = F.tanh(self.critic_fc1(x))
        critic_x = F.tanh(self.critic_fc2(critic_x))
        value = self.critic_fc3(critic_x)

        return policy, value


class A2CAgent:
    def __init__(self, action_size, state_size):
        self.render = True
        self.action_size = action_size
        self.state_size = state_size

        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01

        self.model = A2C(self.action_size, self.state_size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate,weight_decay=0.0005)

    def get_action(self, state):
        # if np.random.rand() <= self.epsilon:
        #     return random.randrange(self.action_size)
        # else:
        policy, _ = self.model(state)
        policy = np.array(policy.detach().cpu())
        print('policy : {}'.format(policy))
        return np.random.choice(self.action_size, 1, p=policy)

    def train_model(self, state, action, reward, next_state, done):
        model_params = self.model.parameters()
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        self.model.train()
        self.optimizer.zero_grad()
        policy, value = self.model(state)
        _, next_value = self.model(next_state)
        next_value2 = next_value.clone()
        next_value2 = next_value2.detach()
        target = reward + (1-done) * self.discount_factor * next_value2

        one_hot_action = Variable(torch.Tensor([int(k==action) for k in range(ACTION_DIM)])).to(device)
        # action_var = Variable(torch.Tensor(action).view(-1, ACTION_DIM))
        print(one_hot_action, policy)
        action_prob = torch.sum(one_hot_action * policy)
        # if target < value:
        #     temp = target
        #     target = value
        #     value = temp

        target2 = target.clone()
        value2 = value.clone()
        # value2 = value2.detach()

        advantage = target2 - value2
        actor_loss = torch.log(action_prob + 1e-5) * advantage
        actor_loss = - torch.mean(actor_loss)

        critic_loss = 0.5 * torch.square((target2 - value2))
        critic_loss = torch.mean(critic_loss)

        loss = 0.2 * actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()
        torch.nn.utils.clip_grad_norm_(model_params, 5.0)
        print('loss : ', loss.detach().cpu().numpy())
        print('target : {}, value : {}'.format(target, value))
        # loss.step()
        return np.array(loss.detach().cpu().numpy())


if __name__ == "__main__":
    env = Env()

    state_size = env.observation_space
    action_size = 2

    agent = A2CAgent(action_size, state_size)

    scores, episodes = [], []
    score_avg = 0

    num_episode = 2000
    max_frame = 500
    # frame = 0
    for e in range(num_episode):
        done = False
        score = 0
        frame = 0
        loss_list = []
        state = env.reset(e)
        state = Variable(torch.Tensor(state)).to(device)

        while not done:

            if agent.render:
                env.render()
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                next_state = Variable(torch.Tensor(next_state)).to(device)

                score += reward
                # reward #= -1 if not done else 100

                loss = agent.train_model(state, action, reward, next_state, done)
                loss_list.append(loss)
                state = next_state

                if done:
                    score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                    log("episode : {} | score_avg : {} | loss : {} | done : {}".format(e, score_avg, np.mean(loss_list), done))
                    torch.save(agent.model.state_dict(), './save_model2/220127_episode_{}.pth'.format(e))
                if frame == 500:
                    score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                    log("episode : {} | score_avg : {} | loss : {} | done : {}".format(e, score_avg, np.mean(loss_list), done))
                    break
                frame += 1
                # plot(frame, score)







