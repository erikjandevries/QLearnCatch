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

from __future__ import division

import logging
log = logging.getLogger(__name__)
from collections import deque;
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np;
import os

from qnet import qnet;


class CatchAgent(object):
    """Agent to act in the Catch environment for the OpenAI Gym"""
    def __init__( self, env
                , memory_maxlen=1000
                , net = None
                , discount = 0.9
                ):
        log.info("Creating an agent to play Catch");
        self.env = env;
        self.memory = deque(maxlen=memory_maxlen);
        if net is None:
            self.net = qnet( (None, self.env.grid_size, self.env.grid_size)
                           , self.env.action_space.n);
        else:
            self.net = net;
        self.discount = discount

    def choose_action(self, observation, policy='random', epsilon=0.1):
        if policy == 'random':
            action = self.env.action_space.sample();
            return action
        elif policy == 'greedy':
            q = self.net.eval_predict(observation[np.newaxis,:,:]);
            action = np.argmax(q[0]);
            return action
        elif policy == 'eps_greedy':
            if np.random.rand() <= epsilon:
                return self.choose_action(observation, policy='random');
            else:
                return self.choose_action(observation, policy='greedy');
        raise ValueError("Policy {} not recognised!".format(policy));

    def store_memory(self, observation_t0, action, reward, observation_t1, done):
        self.memory.append([observation_t0, action, reward, observation_t1, done])

    def get_training_batch(self, batch_size = 50):
        len_memory = len(self.memory)
        inputs = np.zeros((min(len_memory, batch_size),) + self.memory[0][0].shape)
        targets = np.zeros((inputs.shape[0], self.env.action_space.n))
        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            observation_t0, action, reward, observation_t1, done = self.memory[idx]

            inputs[i:i+1] = observation_t0
            # Get current Q-values from the net
            targets[i] = self.net.eval_predict(observation_t0[np.newaxis,:,:])[0]

            # Correct the target Q-value given the observed reward for the given action
            if done:
                targets[i, action] = reward
            else:
                Q_t1 = np.max(self.net.eval_predict(observation_t1[np.newaxis,:,:])[0])
                targets[i, action] = reward + self.discount * Q_t1
        return inputs, targets

    def train(self, n_episodes = 1000, batch_size = 50):
        log.info("Starting training ({} episodes)".format(n_episodes));

        # Keeping track of the number of wins is not needed,
        # but as an indicator of how well our agent is doing,
        # it may be easier to interpret than the value of the loss function.
        wins = 0
        for i_episode in range(n_episodes):
            # The episode starts with resetting the environment.
            observation_t1 = self.env.reset()
            # During training we'll use an epsilon greedy policy to select actions
            # with epsilon decreasing exponentially from 1 to 0.1.
            epsilon = 10**(-i_episode/n_episodes)

            loss = 0.
            done = False
            while not done:
                # The starting point for the current action is the outcome
                # of the previous action:
                observation_t0 = observation_t1;
                # Choose an action using the epsilon greedy policy:
                action = self.choose_action(observation_t0, 'eps_greedy', epsilon);
                # Observe the outcome of taking this action:
                observation_t1, reward, done, info = self.env.step(action);

                # Count the number of wins
                if reward == 1:
                    wins += 1

                # Store the experience in memory
                self.store_memory(observation_t0, action, reward, observation_t1, done)

                # To train the q-net to select better actions in the future:
                # - select a batch of memories to train on
                inputs, targets = self.get_training_batch(batch_size);
                # - train the net
                loss += self.net.eval_train(inputs, targets);

            log.info("Epoch {:03d}/999 | epsilon {:.4f} | Loss {:.4f} | Win count {}".format(i_episode, epsilon, loss, wins))

    def play(self, n_episodes = 10, output_folder = "output"):
        log.info("Playing catch with the trained agent ({} episodes)".format(n_episodes));
        log.info("Output will be saved in the folder: {}".format(output_folder));

        def ensure_dir(d):
            if not os.path.exists(d):
                log.info("Creating folder: {}".format(d));
                os.makedirs(d);
        ensure_dir(output_folder);

        wins = 0
        for e in range(n_episodes):
            # The episode starts with resetting the environment.
            observation = self.env.reset()

            loss = 0.
            done = False

            # save the initial observation as an image
            c = 0
            plt.imshow(self.env.render(mode='rgb_array'), interpolation='none')
            plt.savefig("{}/{:02d}_{:02d}.png".format(output_folder, e, c))
            while not done:
                c += 1
                # get next action
                action = self.choose_action(observation, 'greedy');
                # apply action, get rewards and new state
                observation, reward, done, info = self.env.step(action)

                # Count the number of wins
                if reward == 1:
                    wins += 1

                # save the observation as an image
                plt.imshow(self.env.render(mode='rgb_array'), interpolation='none')
                plt.savefig("{}/{:02d}_{:02d}.png".format(output_folder, e, c))

        log.info("Won {} out of {} games ({} %)".format(wins, n_episodes, (100*wins/n_episodes)));
