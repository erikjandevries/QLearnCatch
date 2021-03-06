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

from __future__ import division;

import logging;
log = logging.getLogger(__name__);
from collections import deque;

# %matplotlib inline
import matplotlib.pyplot as plt;

import numpy as np;

from tools import ensure_dir;
from qnet import qnet;


class CatchAgent(object):
    """Agent to act in the Catch environment for the OpenAI Gym"""
    def __init__( self, env
                , memory_maxlen = 1000
                , net = None
                , discount = 0.9
                , output_folder = "output"
                ):
        log.info("Creating an agent to play Catch");
        self.env = env;
        self.memory = deque(maxlen=memory_maxlen);
        if net is None:
            self.net = qnet( (None, self.env.grid_size, self.env.grid_size)
                           , self.env.action_space.n);
        else:
            self.net = net;
        self.discount = discount;
        ensure_dir(output_folder);
        self.output_folder = output_folder;
        self.results_filename = "{}/results.csv".format(self.output_folder);

    def save_net(self, filename = "net.pkl"):
        self.net.save("{}/{}".format(self.output_folder, filename))

    def load_net(self, filename = "net.pkl"):
        self.net.load("{}/{}".format(self.output_folder, filename))

    def choose_action(self, observation, policy='random', epsilon=0.1):
        if policy == 'random':
            action = self.env.action_space.sample();
            return action
        elif policy == 'greedy':
            q = self.net.predict(observation[np.newaxis,:,:]);
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
            targets[i] = self.net.predict(observation_t0[np.newaxis,:,:])[0]

            # Correct the target Q-value given the observed reward for the given action
            if done:
                targets[i, action] = reward
            else:
                Q_t1 = np.max(self.net.predict(observation_t1[np.newaxis,:,:])[0])
                targets[i, action] = reward + self.discount * Q_t1
        return inputs, targets

    def train( self
             , n_episodes = 1000
             , batch_size = 50
             , render_on_screen = False
             , save_net_every_n_episodes = None
             , learning_rate = None
             ):
        if learning_rate is not None:
            self.net.set_learning_rate(learning_rate);

        log.info("Starting training ({} episodes)".format(n_episodes));

        log.info("Creating the results file: {}".format(self.results_filename));
        with open(self.results_filename, 'w', 0) as results_file:
            results_file.write("episode,epsilon,loss,wins\n");
        try:
            # Keeping track of the number of wins is not needed,
            # but as an indicator of how well our agent is doing,
            # it may be easier to interpret than the value of the loss function.
            wins = 0
            for i_episode in range(n_episodes):
                # During training we'll use an epsilon greedy policy to select actions
                # with epsilon decreasing exponentially from 1 to 0.1.
                epsilon = 10**(-i_episode/n_episodes)

                # The episode starts with resetting the environment.
                observation_t1 = self.env.reset()
                if render_on_screen:
                    self.env.render(mode='human');

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
                    if render_on_screen:
                        self.env.render(mode='human');

                    # Count the number of wins
                    if reward == 1:
                        wins += 1;

                    # Store the experience in memory
                    self.store_memory(observation_t0, action, reward, observation_t1, done)

                    # To train the q-net to select better actions in the future:
                    # - select a batch of memories to train on
                    inputs, targets = self.get_training_batch(batch_size);
                    # - train the net
                    loss += self.net.train(inputs, targets);

                if save_net_every_n_episodes is not None:
                    if i_episode % save_net_every_n_episodes == 0:
                        self.save_net("net/net_{}.pkl".format(i_episode));

                with open(self.results_filename, 'a', 0) as results_file:
                    results_file.write("{},{},{},{}\n".format(i_episode + 1, epsilon, loss, wins));
                log.info("Epoch {:03d}/{} | epsilon {:.4f} | Loss {:.4f} | Win count {}".format(i_episode, n_episodes - 1, epsilon, loss, wins))

        except KeyboardInterrupt:
            print("");
            log.warn('Training interrupted by user');

    def plot_results(self, filename = "results.png"):
        log.info("Plotting results: {}/{}".format(self.output_folder, filename));
        results = np.loadtxt(open(self.results_filename, "r"), delimiter=",", skiprows=1)
        plt.plot(results[:, 0], results[:, 2], '-')
        plt.savefig("{}/{}".format(self.output_folder, filename));
        plt.close();

    def play(self, n_episodes = 10, save_images = True, render_on_screen = False):
        log.info("Playing catch with the trained agent ({} episodes)".format(n_episodes));
        if save_images:
            output_folder = "{}/images".format(self.output_folder);
            ensure_dir(output_folder);
            log.info("Images will be saved in the folder: {}".format(output_folder));

        try:
            wins = 0
            for e in range(n_episodes):
                # The episode starts with resetting the environment.
                observation = self.env.reset();
                if render_on_screen:
                    self.env.render(mode='human');

                loss = 0.;
                done = False;

                c = 0;
                if save_images:
                    # save the initial observation as an image
                    if not render_on_screen:
                        self.env.render(mode='matplotlib');
                    plt.savefig("{}/{:02d}_{:02d}.png".format(output_folder, e, c));
                while not done:
                    c += 1;
                    # get next action
                    action = self.choose_action(observation, 'greedy');
                    # apply action, get rewards and new state
                    observation, reward, done, info = self.env.step(action);
                    if render_on_screen:
                        self.env.render(mode='human');

                    # Count the number of wins
                    if reward == 1:
                        wins += 1;

                    if save_images:
                        # save the observation as an image
                        if not render_on_screen:
                            self.env.render(mode='matplotlib');
                        plt.savefig("{}/{:02d}_{:02d}.png".format(output_folder, e, c))

        except KeyboardInterrupt:
            print("");
            log.warn('Playing interrupted by user');

        log.info("Won {} out of {} games ({} %)".format(wins, n_episodes, (100*wins/n_episodes)));
