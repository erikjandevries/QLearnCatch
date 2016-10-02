"""
   Copyright 2016 Erik Jan de Vries

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import gym;
gym.undo_logger_setup();

import logging;
log = logging.getLogger(__name__);

import numpy as np;
# %matplotlib inline
import matplotlib.pyplot as plt;

from gym import spaces;
from gym.utils import seeding;


class Catch(gym.Env):
    """Catch environment for the OpenAI Gym"""

    metadata = {
        'render.modes': ['human', 'rgb_array', 'matplotlib']
    }

    def __init__(self, grid_size = 10):
        log.info("Creating an OpenAI Gym environment to play Catch");
        log.debug("Grid size: {}".format(grid_size));
        self.grid_size = grid_size;

        self.action_space = spaces.Discrete(3);
        self.observation_space = spaces.Discrete((self.grid_size,self.grid_size));
        self.reward_range = (-1, 1);

        self.seed();
        self.reset();

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed);
        return [seed]

    def _reset(self):
        n = np.random.randint(0, self.grid_size-1, size=1);
        m = np.random.randint(1, self.grid_size-2, size=1);
        self.state = np.asarray([0, n, m]);
        return self._get_observation()

    def _step(self, action):
        if action == 0:
            move = -1      # left
        elif action == 1:
            move = 0       # stay
        else:
            move = 1       # right

        # Get current state
        fruit_row, fruit_col, basket = self.state;
        # Transform state
        fruit_row += 1;
        basket = min(max(1, basket + move), self.grid_size-1);
        # Save new state
        self.state = np.asarray([fruit_row, fruit_col, basket]);

        # Determine the observed new state
        observation = self._get_observation();
        # Determine if we are done
        done = (fruit_row == self.grid_size-1);
        # Determine the reward
        reward = self._get_reward();
        # Set information dictionary
        info = {};

        return observation, reward, done, info

    def _get_observation(self):
        # Get current state
        fruit_row, fruit_col, basket = self.state;
        # Get observation
        observation = np.zeros((self.grid_size, self.grid_size));
        observation[fruit_row, fruit_col] = 1;       # draw the fruit
        observation[-1, (basket-1):(basket+2)] = 1;  # draw the basket
        return observation

    def _get_observation_rgb(self):
        # Get current state
        fruit_row, fruit_col, basket = self.state;
        reward = self._get_reward();
        # Get observation
        observation_rgb = np.zeros((self.grid_size, self.grid_size, 3), dtype='uint8');
        # draw the basket
        observation_rgb[-1, (basket-1):(basket+2), :] = [87, 45, 9];
        # draw the fruit
        if reward == 1:
            observation_rgb[fruit_row, fruit_col, :] = [96, 192, 64];
        elif reward == 0:
            observation_rgb[fruit_row, fruit_col, :] = [192, 192, 64];
        else:
            observation_rgb[fruit_row, fruit_col, :] = [192, 64, 64];
        return observation_rgb

    def _get_reward(self):
        # Get current state
        fruit_row, fruit_col, basket = self.state;
        # Get reward
        if fruit_row == self.grid_size-1:
            if abs(fruit_col - basket) <= 1:
                return 1
            else:
                return -1
        else:
            return 0

    def _plot_observation(self):
        plt.imshow(  self._get_observation_rgb()
                   , interpolation='none'
                  );
        plt.tick_params(
              axis='both'        # changes apply to both the x-axis and the y-axis
            , which='both'       # both major and minor ticks are affected
            , bottom='off'       # ticks  along the bottom edge are off
            , top='off'          # ticks  along the top    edge are off
            , left='off'         # ticks  along the left   edge are off
            , right='off'        # ticks  along the right  edge are off
            , labelbottom='off'  # labels along the bottom edge are off
            , labeltop='off'     # labels along the top    edge are off
            , labelleft='off'    # labels along the left   edge are off
            , labelright='off'   # labels along the right  edge are off
            );

    def _render(self, mode='human', close=False):
        if close:
            if mode == 'human':
                # close all matplotlib screens
                pass
            return
        if mode == 'human':
            self._plot_observation();
            plt.show();
            return
        if mode == 'matplotlib':
            self._plot_observation();
            return
        if mode == 'rgb_array':
            rgb_array = self._get_observation_rgb();
            return rgb_array;
