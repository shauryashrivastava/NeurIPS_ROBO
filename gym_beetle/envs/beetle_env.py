#!/usr/bin/env python

"""
Simulate the robot using models from SINDy and NN

"""

# Core Library
import logging.config
import math
import random
from typing import Any, Dict, List, Tuple
from keras.models import model_from_json

# Third party
import cfg_load
import gym
import numpy as np
import pkg_resources
import math
from gym import spaces

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# remember to check why is this here??? also check the other codes in the banana git
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
path = "config.yaml"  # always use slash in packages
filepath = pkg_resources.resource_filename("gym_beetle", path)
config = cfg_load.load(filepath)
logging.config.dictConfig(config["LOGGING"])


class BeetleEnv(gym.Env):
    """
    Define a simple great_devious_beetle_Env environment.

    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self, params) -> None:
        # self.__version__ = "0.1.0"
        # logging.info(f"BananaEnv - Version {self.__version__}")

        # General variables defining the environment
        # self.MAX_PRICE = 2.0
        # self.TOTAL_TIME_STEPS = 2

        self.curr_step = -1
        # self.is_banana_sold = False

        # Define what the agent can do
        self.action_space = spaces.Box(params.get('ul'), params.get('uh'), shape=(params.get('n_u'),), dtype=np.float32)

        # Observation is the remaining time
        self.observation_space = spaces.Box(params.get('ol'), params.get('oh'), shape=(2,), dtype=np.float32)

        # Store what the agent tried
        self.model=params.get('model')
        self.curr_episode = -1
        self.state=params.get('init_state')
        self.action_episode_memory: List[Any] = []
        self.desiredstate=params.get('desiredstate')
        self.yh=[0,0]

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob : List[int]
                an environment-specific object representing your observation of
                the environment.
            reward : float
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over : bool
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info : Dict
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        ob=self._take_action(action)
        self.curr_step += 1
        reward = self._get_reward()
        episode_over=self.curr_step==199
        return ob, reward, episode_over, {}

    def _take_action(self, action) -> None:
        # self.action_episode_memory[self.curr_episode].append(action)
        s = self.state + action
        self.yh = self.model.predict(s)
        radius=np.sqrt(self.yh[0]**2 + self.yh[1]**2)
        alpha=0
        x=self.yh[0]
        y=self.yh[1]

        if x == 0 and y > 0:
            alpha = 90
        elif x == 0 and y < 0:
            alpha = 270
        elif x > 0 and y == 0:
            alpha = 0
        elif x < 0 and y == 0:
            alpha = 180
        elif x > 0 and y > 0:
            alpha = math.degrees(math.atan(y/x))
        elif x < 0 and y > 0:
            alpha = 180 + math.degrees(math.atan(y/x))
        elif x < 0 and y < 0:
            alpha = 180 + math.degrees(math.atan(y/x))
        elif x > 0 and y < 0:
            alpha = 360 + math.degrees(math.atan(y/x))

        ob= [radius, alpha]
        return ob
 

    def _get_reward(self) -> float:
        """Reward is given for a sold banana."""
        reqposition=self.desiredstate[(self.curr_episode*200)+self.curr_step]
        distance=math.sqrt((reqposition[0]-self.yh[0])**2+(reqposition[1]-self.yh[1])**2)
        reward= math.exp(-distance)
        return reward

    def reset(self) -> List[int]:
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation: List[int]
            The initial observation of the space.
        """
        self.curr_step = -1
        self.curr_episode += 1
        self.action_episode_memory.append([])
        return self._get_state()

    def _render(self, mode: str = "human", close: bool = False) -> None:
        return None

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed
