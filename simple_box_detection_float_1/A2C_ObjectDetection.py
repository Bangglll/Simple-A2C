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

torch.autograd.set_detect_anomaly(True)
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
STATE_DIM = 4
ACTION_DIM = 1 #12
VALUE = 1
hidden_size = 40


def log(logstr):
    file_name = os.path.join('./save_model_2/', '220207.log')
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

class Continuous_A2C(nn.Module): # https://medium.com/deeplearningmadeeasy/advantage-actor-critic-continuous-case-implementation-f55ce5da6b4c
    def __init__ (self, action_size, state_size):
        super(Continuous_A2C, self).__init__()
        self.n_actions = action_size
        self.actor_fc1 = nn.Linear(state_size, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, hidden_size)

        self.actor_mu = nn.Linear(hidden_size, action_size)
        # nn.init.xavier_uniform_(self.actor_mu.weight, 1e-3)
        self.actor_sigma = nn.Linear(hidden_size, action_size)

        logstds_param = nn.Parameter(torch.full((self.n_actions, ),0.1))
        self.register_parameter("logstds", logstds_param)
        # nn.init.xavier_uniform_(self.actor_sigma.weight, 1e-3)

        self.critic_fc1 = nn.Linear(state_size, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, hidden_size)
        self.critic_fc3 = nn.Linear(hidden_size,1)
        # nn.init.xavier_uniform_(self.critic_fc3.weight, 1e-3)

    def forward(self, x):
        actor_x = torch.relu(self.actor_fc1(x))
        actor_x = torch.tanh(self.actor_fc2(actor_x))

        mu = self.actor_mu(actor_x)
        # sigma = torch.sigmoid(self.actor_sigma(actor_x))
        # sigma = sigma + 1e-5
        sigma = torch.clamp(self.logstds.exp(), 1e-3, 50)
        # policy =  F.softmax(self.actor_fc3(actor_x))

        critic_x = torch.tanh(self.critic_fc1(x))
        critic_x = torch.tanh(self.critic_fc2(critic_x))
        value = self.critic_fc3(critic_x)

        return mu, sigma, value


class A2CAgent:
    def __init__(self, action_size, state_size, max_action):
        self.render = True
        self.action_size = action_size
        self.state_size = state_size
        self.max_action = max_action

        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01

        self.model = Continuous_A2C(self.action_size, self.state_size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)#,weight_decay=0.0005)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),2.0)



    def get_action(self, state):
        # if np.random.rand() <= self.epsilon:
        #     return random.randrange(self.action_size)
        # else:
        mu, sigma, _ = self.model(state)
        print('state : {}, mu : {}, sigma : {}'.format(state, mu, sigma))
        dist = torch.distributions.Normal(loc= mu, scale=sigma) #https://pytorch.org/docs/stable/distributions.html#torch.distributions.normal.Normal
        # action_all = dist.sample().clone()
        action = dist.sample().clone()
        action2 = action.detach().cpu().numpy()
        idx = 0 #np.argmax(action2)
        print('action : ', action2)
        # print('action_all : ', action_all)


        action = torch.clip(action, -self.max_action, self.max_action)
        print('action : ', action)
        return action, idx

    def train_model(self, state, action, reward, next_state, done):
        model_params = self.model.parameters()
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        self.model.train()
        self.optimizer.zero_grad()
        mu, sigma, value = self.model(state)

        _, _, next_value = self.model(next_state)
        next_value2 = next_value.clone()
        next_value2 = next_value2.detach()
        target = reward + (1-done) * self.discount_factor * next_value2

        # one_hot_action = Variable(torch.Tensor([int(k==action) for k in range(ACTION_DIM)])).to(device)
        # action_var = Variable(torch.Tensor(action).view(-1, ACTION_DIM))


        # if target < value:
        #     temp = target
        #     target = value
        #     value = temp

        target2 = target.clone()
        # target2 = target2.detach().cpu()
        value2 = value.clone()
        # value2 = value2.detach().cpu()

        advantage = target2 - value2
        dist = torch.distributions.Normal(loc = mu, scale=sigma)
        action_prob = dist.log_prob(action)#torch.sum(one_hot_action * policy)
        # actor_loss = (- action_prob * advantage).mean
        actor_loss = -torch.mean(action_prob*advantage)
        # cross_entropy = torch.log(action_prob + 1e-5) * advantage
        # actor_loss = - torch.mean(cross_entropy )

        critic_loss = F.mse_loss(target, value) #0.5 * torch.square((target2 - value2))
        # critic_loss = torch.mean(critic_loss)

        loss = (0.1 * actor_loss) + critic_loss + 1e-7

        loss.backward()
        nn.utils.clip_grad_norm_([p for g in self.optimizer.param_groups for p in g["params"]], 0.5)
        #nn.utils.clip_grad_norm_([p for g in actor_optim.param_groups for p in g["params"]], max_grad_norm)
        self.optimizer.step()
        # torch.nn.utils.clip_grad_norm_(model_params, 2.0) # 5.0
        print('loss : ', loss.detach().cpu().numpy())
        print('target : {}, value : {}'.format(target, value))
        # loss.step()
        return np.array(loss.detach().cpu().numpy()), sigma.detach().cpu().numpy()


if __name__ == "__main__":
    env = Env()

    state_size = STATE_DIM
    action_size = ACTION_DIM
    max_action = env.high[0]

    agent = A2CAgent(action_size, state_size, max_action)


    scores, episodes = [], []
    score_avg = 0

    num_episode = 2000
    max_frame = 200
    # frame = 0
    for e in range(num_episode):
        done = False
        score = 0
        frame = 0
        loss_list, sigma_list = [], []
        state = env.reset(e)
        state = Variable(torch.Tensor(state)).to(device)

        while not done:

            if agent.render:
                env.render()
                action, idx = agent.get_action(state)
                action2 = action.clone().detach().cpu().numpy()
                print('action2', action2[0])
                action2[0] = round(action2[0], 3)
                next_state, reward, done = env.step(action2, idx)
                next_state = Variable(torch.Tensor(next_state)).to(device)

                score += reward
                # reward #= -1 if not done else 100

                loss, sigma = agent.train_model(state, action, reward, next_state, done)
                loss_list.append(loss)
                sigma_list.append(sigma)
                state = next_state

                # 2200207_episode_1960.pth

                if done:
                    score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                    log("episode : {} | score_avg : {} | loss : {} | sigma : {} | done : {}".format(e, score_avg, np.mean(loss_list), np.mean(sigma_list), done))
                    torch.save(agent.model.state_dict(), './save_model_2/2200207_episode_{}.pth'.format(e))
                if frame == max_frame:
                    score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                    log("episode : {} | score_avg : {} | loss : {} | sigma : {} | done : {}".format(e, score_avg, np.mean(loss_list), np.mean(sigma_list),done))
                    break
                frame += 1
                # plot(frame, score)







