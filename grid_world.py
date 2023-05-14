import gym
from gym import spaces
import pygame
import numpy as np
import pandas as pd
import math

'''
        Incomplete, requires futher testing and debugging
'''

df = pd.read_csv('C4_Complete_Gaze_Trajectory_gaze.csv')
temp = df.to_numpy()[0][0].split(',')
x_0, y_0 = int(temp[1]), int(temp[0])

# 18 vertical lines including border
# 12 horizontal lines including border 
# print(h, w) #720 1152
h_offset = 60
w_offset = 64

# (y, x)
object_points = {
    # (y, x)
    '5,1': 1,
    '5,2': 1,

    '8,4': 2,
    '9,4': 2,

    '9,6': 3,

    '8,6': 4,

    '7,7': 5,

    '6,8': 6,

    '4,8': 7,
    '9,10': 8,
    '5,10': 9,
    '9,11': 10,
    '5,11': 11,
    '5,13': 12

}

def gatherExpert():
    i_obs = pd.read_csv('C4_Complete_Gaze_Trajectory_gaze.csv', usecols= ['point','object', 'ms']).to_numpy()
    i_actions = pd.read_csv('C4_Complete_Gaze_Trajectory_gaze.csv', usecols= ['action'])
    i_states = pd.read_csv('C4_Complete_Gaze_Trajectory_gaze.csv', usecols= ['point']).to_numpy()

    expert = {}
    for i in range(len(i_states)):
        if i_obs[i][0] not in expert:
            expert[i_obs[i][0]] = []
            expert[i_obs[i][0]].append(i_actions.values[i][0])

        elif i_actions.values[i][0] not in expert[i_obs[i][0]]:
            expert[i_obs[i][0]].append(i_actions.values[i][0])

    # print('expert:', expert)
    return expert

def applyGridOffset(a):
    ox = 0
    oy = 0
    x = a[0]
    y = a[1]

    if x / w_offset > 0: ox = x / w_offset
    if y / h_offset > 0: oy = y / h_offset

    # print(x, y, int(round(x/w_offset)), int(round(y/h_offset)))

    return np.array([int(round(ox)), int(round(oy))])

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=15):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        print('my grid world init')
        self._agent_location = np.random.randint(low=0, high=self.size, size=2)
        self._target_location = np.array([x_0, y_0])

        # This commented observation space was for extra.py
        self.observation_space = spaces.Box(0, 15, shape=(4,), dtype=int)

        # We have 9 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(9)

        self._action_to_direction = {
        0: np.array([0, 0]), # stay in current position
        1: np.array([0, 1]), # move up
        2: np.array([-1, 1]), # move up/left diagonal
        3: np.array([-1, 0]), # move left
        4: np.array([-1, -1]), # move down/left diagonal
        5: np.array([0, -1]), # move down
        6: np.array([1, -1]), # move down/right diagonal
        7: np.array([1, 0]), # move right
        8: np.array([1, 1])  # move up/right diagonal
        }

        self.stepCount = 0
        self.object = 0
        self.timestamp = 0

        self.expert = gatherExpert()

        self.isDone = False

        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        # self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        # self.window = None
        # self.clock = None

    def _get_obs(self):
        # return {"agent": self._agent_location, "target": self._target_location}
        # t = []
        # t.append(self._agent_location[1])
        # t.append(self._agent_location[0])
        t = str(self._agent_location[1]) + ',' + str(self._agent_location[0])

        if t in object_points:
            self.object = object_points[t]
        else: 
            self.object = 0

        obs = []
        obs.append(self._agent_location[0])
        obs.append(self._agent_location[1])
        obs.append(self.object)
        obs.append(self.timestamp)

        return np.array(obs)

        # return spaces.Dict(
        #     {
        #         "agent": self._agent_location,
        #         "next_closest": self._target_location,
        #     }
        # )

    def _get_dist(self):
        # print('!!', (if (self._target_location) is None))
        if self._target_location is None: self._target_location = np.array([1, 5])
        # print('inside get info: ', self._agent_location, self._target_location)
        return np.linalg.norm(
                self._agent_location - self._target_location, ord=1
        )

    def _get_info(self):
        # todo
        info = {
            'distance': self._get_dist(),
            'is_success': self.isDone,
            'episode': self.stepCount

        }
        return info

    # sets target location
    def closestTarget(self):
        a = str(self._agent_location[1]) + ',' + str(self._agent_location[0])
        min = self._get_dist()
        target = []
        for pos in self.expert:
            posArr = [int(pos.split(',')[0]), int(pos.split(',')[1])]
            if a != pos:
                temp = np.linalg.norm(self._agent_location - posArr, ord=1)
                if min > temp:
                    self._target_location = posArr
                    min = temp



        # print('Closest target to {} is {}'.format(self._agent_location, self._target_location))

    def consult(self, action):
        self._get_dist()
        a = str(self._agent_location[1]) + ',' + str(self._agent_location[0])
        
        t = []
        isCorrect = False
        if a in self.expert:
            if action in self.expert[a]:
                # Good action, give reward
                self._target_location = self._agent_location + self._action_to_direction[action]
                t = self.expert[a]
                isCorrect = True
                # return True
            
        # print('Consulting Expert: agent{} in experts{}?'.format(action, t))
            
        return isCorrect
            
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = np.random.randint(low=0, high=self.size, size=2)
        # self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = np.array([x_0, y_0])

        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )

        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, info

    def step(self, action):
        # print('stepping with my step')
        prev = self._agent_location
        prevDist = self._get_dist()
        action = math.ceil(action)
        reward = 0

        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Reward for moving in a closer direction and if destination reached
        currDist = self._get_dist()
        if currDist < prevDist: reward += 1
        # else: reward = reward - 1

        if np.array_equal(self._agent_location, self._target_location): 
            self.closestTarget()
            reward += 1

        # Reward for actions taken that expert took
        # reward = 1 if self.consult(action) else 0  # Binary sparse rewards
        observation = self._get_obs()

        if self.stepCount >= 1000:
            terminated = np.bool_(True)
        else: terminated = np.bool_(False)

        info = self._get_info()

        self.stepCount += 1
        self.isDone = terminated

        return observation, reward, self.isDone, info
