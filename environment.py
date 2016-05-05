import gym
import cv2

class Environment:

    def __init__(self, params):
        self.gym = gym.make(params.game)
        self.observation = None
        self.terminal = False
        self.dims = (params.width, params.height)

    def actions(self):
        return self.gym.action_space.n

    def restart(self):
        self.observation = self.gym.reset()
        self.terminal = False

    def act(self, action):
        self.observation, reward, self.terminal, info = self.gym.step(action)
        return reward

    def getScreen(self):
        return cv2.resize(cv2.cvtColor(self.observation, cv2.COLOR_RGB2GRAY), self.dims)

    def isTerminal(self):
        return self.terminal