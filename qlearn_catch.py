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

import gym
gym.undo_logger_setup()

import logging;
logging.basicConfig(  format = '%(asctime)s %(name)s:%(levelname)s: %(message)s'
                    , datefmt='%Y-%m-%d %H:%M:%S'
                    , level  = logging.DEBUG)
log = logging.getLogger(__name__)

# import catch
from gym.envs.registration import register
register(
    id='Catch-v0',
    entry_point='catch:Catch',
    timestep_limit=200,
    reward_threshold=25.0,
)


from agent import CatchAgent;


if __name__ == "__main__":
    env = gym.make('Catch-v0');

    agent = CatchAgent( env
                      , memory_maxlen = 250
                      );

    agent.train( n_episodes = 1000
               , batch_size = 50
               );

    agent.play( n_episodes = 10
              , output_folder = "output"
              );
