import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

PhotoImage = ImageTk.PhotoImage
UNIT = 50  # pixels
HEIGHT = 10  # grid height
WIDTH = 10  # grid width

#np.random.seed(1)

class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)
        self.title('Reinforce')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.counter = 0
        self.rewards = []
        self.goal = []
        # obstacle
        self.set_reward([0, 1], -1)
        self.set_reward([1, np.round(HEIGHT/2)-1], -1)
        self.set_reward([2, HEIGHT-3], -1)
        # #goal
        self.set_reward([WIDTH-1, HEIGHT-1], 1)
        self.set_reward([0, HEIGHT-1], 1)

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        # create grids
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        self.rewards = []
        self.goal = []
        # add image to canvas
        
        self.agents = []
        
        x, y = (WIDTH-1)*UNIT - UNIT / 2, UNIT/2 
        agent1 = canvas.create_image(1.5*UNIT, y, image=self.shapes[0])
        agent2 = canvas.create_image(x, y, image=self.shapes[0])
        self.agents.append(agent1)
        self.agents.append(agent2)

        # pack all`
        canvas.pack()

        return canvas

    def load_images(self):
        rectangle = PhotoImage(
            Image.open("img/rectangle.png").resize((UNIT-20, UNIT-20)))
        triangle = PhotoImage(
            Image.open("img/triangle.png").resize((UNIT-20, UNIT-20)))
        circle = PhotoImage(
            Image.open("img/circle.png").resize((UNIT-20, UNIT-20)))

        return rectangle, triangle, circle

    def reset_reward(self):

        for reward in self.rewards:
            self.canvas.delete(reward['figure'])

        self.rewards.clear()
        self.goal.clear()
        self.set_reward([0, 1], -1)
        self.set_reward([1, np.round(HEIGHT/2)-1], -1)
        self.set_reward([2, HEIGHT-3], -1)

        #goal
        self.set_reward([WIDTH-1, HEIGHT-1], 1)
        self.set_reward([0, HEIGHT-1], 1)

    def set_reward(self, state, reward):
        state = [int(state[0]), int(state[1])]
        x = int(state[0])
        y = int(state[1])
        temp = {}
        if reward > 0:
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                       (UNIT * y) + UNIT / 2,
                                                       image=self.shapes[2])
            
            temp['coords'] = self.canvas.coords(temp['figure'])
            temp['state'] = state
            self.goal.append(temp)


        elif reward < 0:
            temp['direction'] = -1
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                      (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[1])
            
            temp['coords'] = self.canvas.coords(temp['figure'])
            temp['state'] = state
            self.rewards.append(temp)

    # new methods

    def check_if_reward(self, state, agentNr):
        check_list = dict()
        check_list['if_goal'] = False
        rewards = 0

        for reward in self.rewards:
            if reward['state'] == state:
                rewards += reward['reward']
        
        for goal in self.goal:
            if goal['state'] == state:
                if self.goal.index(goal) == agentNr:
                    rewards += goal['reward']
                    check_list['if_goal'] = True
        
        for agent in self.agents:
            if self.agents.index(agent) == agentNr:
                continue
            if self.coords_to_state(self.canvas.coords(agent)) == state:
                rewards -= 0.5
        check_list['rewards'] = rewards

        return check_list

    def coords_to_state(self, coords):
        x = int((coords[0] - UNIT / 2) / UNIT)
        y = int((coords[1] - UNIT / 2) / UNIT)
        return [x, y]

    def reset(self, agentNr):
        self.update()
        x, y = self.canvas.coords(self.agents[agentNr])

        # return observation
        if agentNr == 0:
            self.canvas.move(self.agents[agentNr], 1.5*UNIT - x, UNIT / 2 - y)
        else:
            self.canvas.move(self.agents[agentNr], (WIDTH-1)*UNIT - UNIT/2 - x, UNIT / 2 - y)
        
        self.reset_reward()
        return self.get_state(agentNr)

    def step(self, action, agentNr):
        self.counter += 1
        self.render()

        if self.counter % 2 == 1:
            self.rewards = self.move_rewards()

        next_coords = self.move(self.agents[agentNr], action)
        check = self.check_if_reward(self.coords_to_state(next_coords), agentNr)
        done = check['if_goal']
        reward = check['rewards']
        reward -= 0.1
        self.canvas.tag_raise(self.agents[0])
        self.canvas.tag_raise(self.agents[1])

        s_ = self.get_state(agentNr)

        return s_, reward, done

    def get_state(self, agentNr):

        location1 = self.coords_to_state(self.canvas.coords(self.agents[agentNr]))
        agent_x1 = location1[0]
        agent_y1 = location1[1]

        states = list()

        # locations.append(agent_x)
        # locations.append(agent_y)

        for reward in self.rewards:
            reward_location = reward['state']
            states.append(reward_location[0] - agent_x1)
            states.append(reward_location[1] - agent_y1)
            states.append(-1)
            states.append(reward['direction'])
                
        for goal in self.goal:
            if self.goal.index(goal) == agentNr:
                reward_location = goal['state']
                states.append(reward_location[0] - agent_x1)
                states.append(reward_location[1] - agent_y1)
                states.append(1)
                
        for agent in self.agents:
            if self.agents.index(agent) == agentNr:
                continue
            location = self.coords_to_state(self.canvas.coords(agent))
            states.append(location[0] - agent_x1)
            states.append(location[1] - agent_y1)
            states.append(-0.5)
            
        return states

    def move_rewards(self):
        new_rewards = []
        for temp in self.rewards:
            if temp['reward'] > 0:
                new_rewards.append(temp)
                continue
            temp['coords'] = self.move_const(temp)
            temp['state'] = self.coords_to_state(temp['coords'])
            new_rewards.append(temp)
            self.update()
        return new_rewards

    def move_const(self, target):

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

        if (target['figure'] not in self.agents
           and s == [(WIDTH - 1) * UNIT, (HEIGHT - 1) * UNIT]):
            base_action = np.array([0, 0])

        self.canvas.move(target['figure'], base_action[0], base_action[1])

        s_ = self.canvas.coords(target['figure'])

        return s_

    def move(self, target, action):
        s = self.canvas.coords(target)

        base_action = np.array([0, 0])

        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(target, base_action[0], base_action[1])

        s_ = self.canvas.coords(target)

        return s_

    def render(self):
        time.sleep(0.001)
        self.update()