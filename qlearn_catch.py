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
logging.basicConfig(  format = '%(asctime)s %(name)s:%(levelname)s: %(message)s'
                    , datefmt='%Y-%m-%d %H:%M:%S'
                    , level  = logging.DEBUG);
log = logging.getLogger(__name__);

### Register the game Catch at the gym
from gym.envs.registration import register;
register(
    id='Catch-v0',
    entry_point='catch:Catch',
    kwargs={'grid_size' : 10},
    timestep_limit=200,
    reward_threshold=25.0,
);


from agent import CatchAgent;


if __name__ == "__main__":
    ### Create the environment
    env = gym.make('Catch-v0');

    ### Create the agent
    agent = CatchAgent( env
                      , memory_maxlen = 250
                      , output_folder = "output"
                      );

    # ### Load saved net
    # agent.net.load("output/net.pkl");

    ### Train the agent
    agent.train( n_episodes = 1000
               , batch_size = 50
               , learning_rate = 0.5
               , save_net_every_n_episodes = 100
               , render_on_screen = False
               );
    agent.save_net("net.pkl");
    agent.plot_results("results.png");

    ### Play the game
    # agent.net.load("output/net/net.pkl");
    agent.play( n_episodes = 10
              , save_images = True
            #   , render_on_screen = True
              );
