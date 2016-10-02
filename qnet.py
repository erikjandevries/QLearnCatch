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

import logging;
log = logging.getLogger(__name__);

import cPickle as pickle;
import numpy as np;
import os;

import theano;
import theano.tensor as T;
import lasagne;
from lasagne.layers import InputLayer;
from lasagne.layers import DenseLayer;
from lasagne.utils import floatX;

from tools import ensure_dir;


class qnet(object):
    """Neural network for estimating the Q-value function"""
    def __init__( self
                , input_shape
                , num_actions
                , input_var     = None
                , hidden_size   = 100
                , learning_rate = 0.1
                ):
        log.debug("Creating qnet");
        self.net = self.build_net(input_shape, input_var, hidden_size, num_actions);

        input_var = self.net['input'].input_var;
        target_var = T.matrix('targets');

        log.debug("Defining and compiling a Theano function to predict Q-values");
        prediction = lasagne.layers.get_output(self.net['output']);
        self.tf_predict = theano.function([input_var], prediction);

        log.debug("Defining the loss function for training");
        # Create a loss expression for training
        # i.e. a scalar objective we want to minimize
        loss = lasagne.objectives.squared_error(prediction, target_var);
        loss = loss.mean();

        # Define a shared variable for the learning rate
        self.learning_rate = theano.shared( np.array(learning_rate, dtype=theano.config.floatX)
                                          , 'learning_rate'); # 0.113

        # Create update expressions for training
        # i.e. how to modify the parameters at each training step.
        # Here, we'll use Stochastic Gradient Descent (SGD).
        params = lasagne.layers.get_all_params(self.net['output'], trainable=True);
        updates = lasagne.updates.sgd(loss, params, learning_rate=0.2);

        log.debug("Compiling the Theano training function");
        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        self.tf_train = theano.function([input_var, target_var], loss, updates=updates);

    def build_net(self, input_shape, input_var, hidden_size, num_actions):
        log.debug("Building deep neural network")
        net = {};
        net['input']  = InputLayer(input_shape, input_var=input_var);
        net['dense1'] = DenseLayer(net['input'], num_units=hidden_size);
        net['dense2'] = DenseLayer(net['dense1'], num_units=hidden_size);
        net['output'] = DenseLayer(net['dense2'], num_units=num_actions);
        return net

    def set_learning_rate(self, learning_rate):
        log.debug("Setting learning_rate: {}".format(learning_rate));
        self.learning_rate.set_value(np.array(learning_rate, dtype=theano.config.floatX));

    def predict(self, batch_input):
        return self.tf_predict(floatX(batch_input))

    def train(self, batch_input, batch_target):
        return self.tf_train(floatX(batch_input), floatX(batch_target));

    def save(self, net_file):
        ensure_dir(os.path.dirname(net_file));
        log.info("Saving net file: {}".format(net_file));
        with open(net_file, 'wb') as pkl_file:
            param_values = lasagne.layers.get_all_param_values(self.net['output']);
            pickle.dump(param_values, pkl_file);

    def load(self, net_file):
        log.info("Loading net file: {}".format(net_file));
        with open(net_file, 'rb') as pkl_file:
            param_values = pickle.load(pkl_file);
            lasagne.layers.set_all_param_values(self.net['output'], param_values);
