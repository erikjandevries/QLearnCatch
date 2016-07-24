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

import logging
log = logging.getLogger(__name__)
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.utils import floatX;


class qnet(object):
    """Neural network for estimating the Q-value function"""
    def __init__(self, input_shape, num_actions, input_var=None, hidden_size=100):
        log.debug("Creating qnet");

        target_var = T.matrix('targets');

        self.net = {}
        self.net['input']  = InputLayer(input_shape, input_var=input_var)
        self.net['dense1'] = DenseLayer(self.net['input'], num_units=hidden_size)
        self.net['dense2'] = DenseLayer(self.net['dense1'], num_units=hidden_size)
        self.net['output'] = DenseLayer(self.net['dense2'], num_units=num_actions)

        input_var = self.net['input'].input_var;

        log.debug("Defining and compiling a Theano function to predict Q-values");
        prediction = lasagne.layers.get_output(self.net['output']);
        self.predict = theano.function([input_var], prediction);

        log.debug("Defining the loss function for training");
        # Create a loss expression for training
        # i.e. a scalar objective we want to minimize
        loss = lasagne.objectives.squared_error(prediction, target_var)
        loss = loss.mean()

        # Create update expressions for training
        # i.e. how to modify the parameters at each training step.
        # Here, we'll use Stochastic Gradient Descent (SGD).
        params = lasagne.layers.get_all_params(self.net['output'], trainable=True)
        updates = lasagne.updates.sgd(loss, params, learning_rate=0.2)

        log.debug("Compiling the Theano training function")
        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        self.train = theano.function([input_var, target_var], loss, updates=updates)

    def prep_inp(self, inp):
        return floatX(inp)

    def eval_predict(self, inp):
        return self.predict(self.prep_inp(inp))

    def eval_train(self, inp, trg):
        return self.train(self.prep_inp(inp), floatX(trg))
