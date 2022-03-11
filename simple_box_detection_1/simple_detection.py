import time
import numpy as np
import tkinter as tk
from tkinter import Button
from PIL import ImageTk, Image
from random import *

PhotoImage = ImageTk.PhotoImage
UNIT = 50  # 픽셀 수
HEIGHT = 10  # 그리드월드 세로
WIDTH = 10  # 그리드월드 가로

np.random.seed(1)

# action space
# 0 : Push Box to the left
# 1 : Push Box to the right

# Observation (state)
# [0: x1, 1: y1, 2: x2, 3: y2]

class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['l', 'r']
        self.action_size = len(self.action_space)
        self.title('Reinforce')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        # self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.observation_space = 4#[0, 0, 0, 0]
        self.state = None
        self.counter = 0
        self.false_iou = 0
        self.reward = 0
        self.rewards = []
        self.goal = []

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)

        canvas.create_line(0, HEIGHT/2*UNIT, WIDTH*UNIT, HEIGHT/2*UNIT, width=2, fill='blue')
        self.episode_text = canvas.create_text(40, 20, text="episode = {}".format(0), font=("나눔고딕코딩", 10),
                                               fill="blue")
        self.gt_box = canvas.create_rectangle(WIDTH/2*UNIT-UNIT, HEIGHT/2*UNIT-HEIGHT/4*UNIT,WIDTH/2*UNIT+UNIT, HEIGHT/2*UNIT+HEIGHT/4*UNIT, width=2, outline='red')
        print('{}, {}, {}, {}'.format(WIDTH/2*UNIT-UNIT, HEIGHT/2*UNIT-HEIGHT/4*UNIT,WIDTH/2*UNIT+UNIT, HEIGHT/2*UNIT+HEIGHT/4*UNIT))
        x1 = randint(0,400)
        self.r1 = canvas.create_rectangle(x1, HEIGHT/2*UNIT-HEIGHT/4*UNIT,x1+100, HEIGHT/2*UNIT+HEIGHT/4*UNIT, width=2, outline='green')
        print('{}, {}, {}, {}'.format(x1, HEIGHT/2*UNIT-HEIGHT/4*UNIT,x1+100, HEIGHT/2*UNIT+HEIGHT/4*UNIT))

        self.rewards = []
        self.goal = []

        canvas.pack()


        return canvas

    def reset_reward(self):

        for reward in self.rewards:
            self.canvas.delete(reward['figure'])

        self.rewards.clear()
        self.goal.clear()

        gt_box_coord = self.canvas.coords(self.gt_box)
        # self.set_reward([gt_box_coord[0], gt_box_coord[1], gt_box_coord[2], gt_box_coord[3]], 1) #

    def IoU(self, state):
        # box = (x1, y1, x2, y2)
        box1 = self.canvas.coords(self.gt_box)
        box2 = self.canvas.coords(self.r1)
        # box2 = state
        print('state', state)
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # obtain x1, y1, x2, y2 of the intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # compute the width and height of the intersection
        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)

        inter = w * h
        iou = inter / (box1_area + box2_area - inter)
        print(iou)
        return iou

    def set_reward(self, state, reward):
        state = [int(state[0]), int(state[1])]
        x = int(state[0])
        y = int(state[1])
        temp = {}


    def check_if_reward(self, state):
        check_list = dict()
        check_list['if_goal'] = False
        rewards = 0
        iou = self.IoU(state)

        r1_coord = self.canvas.coords(self.r1)

        if iou > 0 and iou < 0.7:
            rewards+= 0.5
        elif iou >= 0.7 and iou < 0.9:
            rewards+= 1
        elif iou >=0.9:
            rewards+= 10
            check_list['if_goal'] = True
        elif r1_coord[2] > WIDTH * UNIT or r1_coord[0] < 0:
            self.false_iou += 1
            rewards -= 1
        elif iou == 0:
            self.false_iou += 1
            rewards -= 1
        else:
            rewards -= 1#(1.0 - iou)
        if self.false_iou == 500:
            rewards -= 100
            # check_list['if_goal'] = True

        check_list['rewards'] = rewards

        return check_list

    def coords_to_state(self, coords): #필요 없어
        x = int((coords[0] - UNIT / 2) / UNIT)
        y = int((coords[1] - UNIT / 2) / UNIT)
        return [x, y]

    def keypress(self,event):
        """The 4 key press"""
        r1_coord = self.canvas.coords(self.r1)
        x, y = 0, 0
        if event.char == "a":
            x = -1
        if event.char == "d":
            x = 1

        if r1_coord[0] + x < 1 or r1_coord[2] + x > WIDTH * UNIT - 1:
            x = 0
        self.canvas.move(self.r1, x, y)
        print(self.canvas.coords(self.r1))
        iou = self.IoU()
        if iou == 1:
            self.reset()

    def rand_move_coord(self):

        x1 = randint(-500, 500)
        return x1

    def reset(self, episode):
        self.update()
        self.canvas.delete(self.episode_text)
        self.episode_text = self.canvas.create_text(40, 20, text="episode = {}".format(episode), font=("나눔고딕코딩", 10),
                                               fill="blue")
        self.false_iou = 0
        r1_coord = self.canvas.coords(self.r1)
        # while(r1_coord[0]+x1+100>WIDTH*UNIT and r1_coord[0]+x1<0):
        while(1):
            x1 = self.rand_move_coord()
            if r1_coord[2] + x1 > WIDTH * UNIT or r1_coord[0] + x1 < 0:
                x1 = self.rand_move_coord()
            else:
                break
        self.canvas.move(self.r1, x1, 0)
        self.reset_reward()
        self.reward = 0
        s = self.canvas.coords(self.r1)
        # return s
        return self.get_state()

    def step(self, action): #
        self.counter += 1
        self.render()
        gt_coord = self.canvas.coords(self.gt_box)
        # x1, y1, x2, y2 = self.state

        # if self.counter % 2 == 1:
        #     self.rewards = self.move_rewards()
        print("action : ", action)
        next_coords = self.move_rect(action)
        # next_coords = self.move_rect_float(action)
        print('next_coords', next_coords)
        check = self.check_if_reward(next_coords)
        print('check', check)
        done = check['if_goal']
        reward = check['rewards']
        # reward -= 0.1
        # self.reward -= 0.1
        # self.canvas.tag_raise(self.rectangle)

        # s_ = self.canvas.coords(self.r1)
        # self.state = (s_[0], s_[1], s_[2], s_[3])
        s_ = self.get_state()

        return s_, reward, done

    def get_state(self): #

        location = self.canvas.coords(self.r1)
        agent_x1 = location[0]
        agent_y1 = location[1]
        agent_x2 = location[2]
        agent_y2 = location[3]

        gt_box_coord = self.canvas.coords(self.gt_box)

        states = list()

        # for reward in self.rewards:
        #     reward_location = reward['state']
        states.append(gt_box_coord[0] - agent_x1)
        states.append(gt_box_coord[1] - agent_y1)
        states.append(gt_box_coord[2] - agent_x2)
        states.append(gt_box_coord[3] - agent_y2)
            # if reward['reward'] < 0:
            #     states.append(-1)
            #     states.append(reward['direction'])
            # else:
            #     states.append(1)

        return states

    def move_rewards(self): #
        new_rewards = []
        for temp in self.rewards:
            if temp['reward'] > 0:
                new_rewards.append(temp)
                continue
            temp['coords'] = self.move_const(temp)
            temp['state'] = self.coords_to_state(temp['coords'])
            new_rewards.append(temp)
        return new_rewards

    def move_const(self, target): #

        s = self.canvas.coords(target['figure'])

        base_action = np.array([0, 0])

        if s[0] == (WIDTH - 1) * UNIT + UNIT / 2:
            target['direction'] = 1
        elif s[0] == UNIT / 2:
            target['direction'] = -1

        if target['direction'] == -1:
            base_action[0] += UNIT
        elif target['direction'] == 1:
            base_action[0] -= UNIT

        if (target['figure'] is not self.rectangle
           and s == [(WIDTH - 1) * UNIT, (HEIGHT - 1) * UNIT]):
            base_action = np.array([0, 0])

        self.canvas.move(target['figure'], base_action[0], base_action[1])

        s_ = self.canvas.coords(target['figure'])

        return s_

    def move_rect(self, action):
        act = action
        # if action[0] > action[1]:
        #     act = 0
        # else:
        #     act = 1
        s = self.canvas.coords(self.r1)
        x, y = 0, 0
        if act == 0: #좌
            if s[0] > 0:
                x = -10
        if act == 1: # 우
            if s[2] < WIDTH*UNIT :
                x = 10

        self.canvas.move(self.r1, x, y)
        # canvas.move(t, x, y)
        print(self.canvas.coords(self.r1))
        s_ = self.canvas.coords(self.r1)
        return s_

    def move_rect_float(self, action):
        # if action[0] > action[1]:
        #     act = 0
        # else:
        #     act = 1
        s = self.canvas.coords(self.r1)
        x, y = 0, 0
        if s[0] >0 or s[1] < WIDTH*UNIT :
            x = action
        # if act == 0: #좌
        #     if s[0] > 0:
        #         x = -10
        # if act == 1: # 우
        #     if s[2] < WIDTH*UNIT :
        #         x = 10

        self.canvas.move(self.r1, x, y)
        # canvas.move(t, x, y)
        print(self.canvas.coords(self.r1))
        s_ = self.canvas.coords(self.r1)
        return s_

    def render(self):
        # 게임 속도 조정
        time.sleep(0.07)
        self.update()

if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env()
    env.bind("<Key>", env.keypress)
    env.mainloop()

