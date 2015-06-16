#!/usr/bin/env python
import logging
from argparse import ArgumentParser

import theano
from theano import tensor
import theano.tensor as T

import blocks
from blocks.algorithms import GradientDescent, Adam
from blocks.bricks import MLP, Tanh, WEIGHT, Rectifier
from blocks.initialization import Constant, Sparse, Orthogonal
from fuel.streams import DataStream
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme, ShuffledScheme
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Checkpoint, Load
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.plot import Plot
from blocks.main_loop import MainLoop

from blocks.bricks.cost import BinaryCrossEntropy
from blocks.bricks import Logistic
import fuel
import os
from fuel.datasets.hdf5 import H5PYDataset
floatX = theano.config.floatX
import numpy as np
import cPickle as pickle
from fuel.transformers import Flatten

import sys
if sys.gettrace() is not None:
    print "Debugging"
    theano.config.optimizer='fast_compile' #"None"  #
    theano.config.exception_verbosity='high'
    theano.config.compute_test_value = 'warn'

#-----------------------------------------------------------------------------
from blocks.bricks import Initializable, Random, Linear
from blocks.bricks.base import application

class Qlinear(Initializable):
    """
    brick to handle the intermediate layer of an Autoencoder.
    In this brick a simple linear mix is performed (a kind of PCA.)
    """
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Qlinear, self).__init__(**kwargs)

        self.mean_transform = Linear(
                name=self.name+'_mean',
                input_dim=input_dim, output_dim=output_dim,
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.children = [self.mean_transform]

    def get_dim(self, name):
        if name == 'input':
            return self.mean_transform.get_dim('input')
        elif name == 'output':
            return self.mean_transform.get_dim('output')
        else:
            raise ValueError

    @application(inputs=['x'], outputs=['z', 'kl_term'])
    def sample(self, x):
        """Sampling is trivial in this case
        """
        mean = self.mean_transform.apply(x)

        z = mean

        # Calculate KL
        batch_size = x.shape[0]
        kl = T.zeros((batch_size,),dtype=floatX)

        return z, kl

    @application(inputs=['x'], outputs=['z'])
    def mean_z(self, x):
        return self.mean_transform.apply(x)


class Qsampler(Qlinear, Random):
    """
    brick to handle the intermediate layer of an Autoencoder.
    The intermidate layer predict the mean and std of each dimension
    of the intermediate layer and then sample from a normal distribution.
    """
    # Special brick to handle Variatonal Autoencoder statistical sampling
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Qsampler, self).__init__(input_dim, output_dim, **kwargs)

        self.prior_mean = 0.
        self.prior_log_sigma = 0.

        self.log_sigma_transform = Linear(
                name=self.name+'_log_sigma',
                input_dim=input_dim, output_dim=output_dim,
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.children.append(self.log_sigma_transform)

    @application(inputs=['x'], outputs=['z', 'kl_term'])
    def sample(self, x):
        """Return a samples and the corresponding KL term

        Parameters
        ----------
        x :

        Returns
        -------
        z : tensor.matrix
            Samples drawn from Q(z|x)
        kl : tensor.vector
            KL(Q(z|x) || P_z)

        """
        mean = self.mean_transform.apply(x)
        log_sigma = self.log_sigma_transform.apply(x)

        batch_size = x.shape[0]
        dim_z = self.get_dim('output')

        # Sample from mean-zeros std.-one Gaussian
        u = self.theano_rng.normal(
                    size=(batch_size, dim_z),
                    avg=0., std=1.)
        z = mean + tensor.exp(log_sigma) * u

        # Calculate KL
        kl = (
            self.prior_log_sigma - log_sigma
            + 0.5 * (
                tensor.exp(2 * log_sigma) + (mean - self.prior_mean) ** 2
                ) / tensor.exp(2 * self.prior_log_sigma)
            - 0.5
        ).sum(axis=-1)

        return z, kl
#-----------------------------------------------------------------------------


class VAEModel(Initializable):
    """
    A brick to perform the entire auto-encoding process
    """
    def __init__(self,
                    encoder_mlp, sampler,
                    decoder_mlp, **kwargs):
        super(VAEModel, self).__init__(**kwargs)

        self.encoder_mlp = encoder_mlp
        self.sampler = sampler
        self.decoder_mlp = decoder_mlp

        self.children = [self.encoder_mlp, self.sampler, self.decoder_mlp]

    def get_dim(self, name):
        if name in ['z', 'z_mean', 'z_log_sigma']:
            return self.sampler.get_dim('output')
        elif name == 'kl':
            return 0
        else:
            super(VAEModel, self).get_dim(name)

    @application(inputs=['features'], outputs=['reconstruction', 'kl_term'])
    def reconstruct(self, features):
        enc = self.encoder_mlp.apply(features)
        z, kl = self.sampler.sample(enc)

        x_recons = self.decoder_mlp.apply(z)
        x_recons.name = "reconstruction"

        kl.name = "kl"

        return x_recons, kl

    @application(inputs=['features'], outputs=['z', 'enc'])
    def mean_z(self, features):
        enc = self.encoder_mlp.apply(features)
        z = self.sampler.mean_z(enc)

        return z, enc

#-----------------------------------------------------------------------------

def shnum(value):
    """ Convert a float into a short tag-usable string representation. E.g.:
        <=0 -> 0
        0.1   -> 11
        0.01  -> 12
        0.001 -> 13
        0.005 -> 53
    """
    if value <= 0.:
        return '0'
    exp = np.floor(np.log10(value))
    leading = ("%e"%value)[0]
    return "%s%d" % (leading, -exp)

def main(name, model, epochs, batch_size, learning_rate, bokeh, layers, gamma,
         rectifier, predict, dropout, qlinear, sparse):
    runname = "vae%s-L%s%s%s%s-l%s-g%s-b%d" % (name, layers,
                                            'r' if rectifier else '',
                                            'd' if dropout else '',
                                            'l' if qlinear else '',
                                      shnum(learning_rate), shnum(gamma), batch_size//100)
    if rectifier:
        activation = Rectifier()
        full_weights_init = Orthogonal()
    else:
        activation = Tanh()
        full_weights_init = Orthogonal()

    if sparse:
        runname += '-s%d'%sparse
        weights_init = Sparse(num_init=sparse, weights_init=full_weights_init)
    else:
        weights_init = full_weights_init

    layers = map(int,layers.split(','))

    encoder_layers = layers[:-1]
    encoder_mlp = MLP([activation] * (len(encoder_layers)-1),
              encoder_layers,
              name="MLP_enc", biases_init=Constant(0.), weights_init=weights_init)

    enc_dim = encoder_layers[-1]
    z_dim = layers[-1]
    if qlinear:
        sampler = Qlinear(input_dim=enc_dim, output_dim=z_dim, biases_init=Constant(0.), weights_init=full_weights_init)
    else:
        sampler = Qsampler(input_dim=enc_dim, output_dim=z_dim, biases_init=Constant(0.), weights_init=full_weights_init)

    decoder_layers = layers[:]  ## includes z_dim as first layer
    decoder_layers.reverse()
    decoder_mlp = MLP([activation] * (len(decoder_layers)-2) + [Logistic()],
              decoder_layers,
              name="MLP_dec", biases_init=Constant(0.), weights_init=weights_init)


    vae = VAEModel(encoder_mlp, sampler, decoder_mlp)
    vae.initialize()

    x = tensor.matrix('features')/256.
    x.tag.test_value = np.random.random((batch_size,layers[0])).astype(np.float32)

    if predict:
        mean_z, enc = vae.mean_z(x)
        # cg = ComputationGraph([mean_z, enc])
        newmodel = Model([mean_z,enc])
    else:
        x_recons, kl_terms = vae.reconstruct(x)
        recons_term = BinaryCrossEntropy().apply(x, x_recons)
        recons_term.name = "recons_term"

        cost = recons_term + kl_terms.mean()
        cg = ComputationGraph([cost])

        if gamma > 0:
            weights = VariableFilter(roles=[WEIGHT])(cg.variables)
            cost += gamma * blocks.theano_expressions.l2_norm(weights)

        cost.name = "nll_bound"
        newmodel = Model(cost)

        if dropout:
            from blocks.roles import INPUT
            inputs = VariableFilter(roles=[INPUT])(cg.variables)
            # dropout_target = [v for k,v in newmodel.get_params().iteritems()
            #            if k.find('MLP')>=0 and k.endswith('.W') and not k.endswith('MLP_enc/linear_0.W')]
            dropout_target = filter(lambda x: x.name.startswith('linear_'), inputs)
            cg = apply_dropout(cg, dropout_target, 0.5)
            target_cost = cg.outputs[0]
        else:
            target_cost = cost

    if name == 'mnist':
        if predict:
            train_ds = MNIST("train")
        else:
            train_ds = MNIST("train", sources=['features'])
        test_ds = MNIST("test")
    else:
        datasource_dir = os.path.join(fuel.config.data_path, name)
        datasource_fname = os.path.join(datasource_dir , name+'.hdf5')
        if predict:
            train_ds = H5PYDataset(datasource_fname, which_set='train')
        else:
            train_ds = H5PYDataset(datasource_fname, which_set='train', sources=['features'])
        test_ds = H5PYDataset(datasource_fname, which_set='test')
    train_s = Flatten(DataStream(train_ds,
                 iteration_scheme=ShuffledScheme(
                     train_ds.num_examples, batch_size)))
    test_s = Flatten(DataStream(test_ds,
                 iteration_scheme=ShuffledScheme(
                     test_ds.num_examples, batch_size)))

    if predict:
        from itertools import chain
        fprop = newmodel.get_theano_function()
        allpdata = None
        alledata = None
        f = train_s.sources.index('features')
        assert f == test_s.sources.index('features')
        sources = test_s.sources
        alllabels = dict((s,[]) for s in sources if s != 'features')
        for data in chain(train_s.get_epoch_iterator(), test_s.get_epoch_iterator()):
            for s,d in zip(sources,data):
                if s != 'features':
                    alllabels[s].extend(list(d))

            pdata, edata = fprop(data[f])
            if allpdata is None:
                allpdata = pdata
            else:
                allpdata = np.vstack((allpdata, pdata))
            if alledata is None:
                alledata = edata
            else:
                alledata = np.vstack((alledata, edata))
        print 'Saving',allpdata.shape,'intermidiate layer, for all training and test examples, to',name+'_z.npy'
        np.save(name+'_z', allpdata)
        print 'Saving',alledata.shape,'last encoder layer to',name+'_e.npy'
        np.save(name+'_e', alledata)
        print 'Saving additional labels/targets:',','.join(alllabels.keys()),
        print ' of size',','.join(map(lambda x: str(len(x)),alllabels.values())),
        print 'to',name+'_labels.pkl'
        with open(name+'_labels.pkl','wb') as fp:
            pickle.dump(alllabels, fp, -1)
    else:
        cg = ComputationGraph([target_cost])
        algorithm = GradientDescent(
            cost=target_cost, params=cg.parameters,
            step_rule=Adam(learning_rate)  # Scale(learning_rate=learning_rate)
        )
        extensions = []
        if model:
            extensions.append(Load(model))

        extensions += [Timing(),
                      FinishAfter(after_n_epochs=epochs),
                      DataStreamMonitoring(
                          [cost, recons_term],
                          test_s,
                          prefix="test"),
                      TrainingDataMonitoring(
                          [cost,
                           aggregation.mean(algorithm.total_gradient_norm)],
                          prefix="train",
                          after_epoch=True),
                      Checkpoint(runname, every_n_epochs=10),
                      Printing()]

        if bokeh:
            extensions.append(Plot(
                'Auto',
                channels=[
                    ['test_recons_term','test_nll_bound','train_nll_bound'
                     ],
                    ['train_total_gradient_norm']]))

        main_loop = MainLoop(
            algorithm,
            train_s,
            model=newmodel,
            extensions=extensions)

        main_loop.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training a Variational-Autoencoder.")
    parser.add_argument("--name", default="mnist",
                        help="name of hdf5 data set")
    parser.add_argument("--model",
                        help="start model to read")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of training epochs to do.")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                default=500, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                default=1e-3, help="Learning rate")
    parser.add_argument("--bokeh", action='store_true', default=False,
                        help="Set if you want to use Bokeh ")
    parser.add_argument("--layers",
                default="784,100,20", help="number of units in each layer of the encoder"
                                           " (use 784, on first layer, for mnist.)"
                                           " The last number (e.g. 20) is the dimension of the intermidiate layer."
                                           " The decoder has the same layers as the encoder but in reverse"
                                           " (e.g. 100, 784)")
    parser.add_argument("--gamma", type=float,
                default=3e-4, help="L2 weight")
    parser.add_argument("-r","--rectifier",action='store_true',default=False,
                        help="Use RELU activation on hidden (default Tanh)")
    parser.add_argument("-p","--predict",action='store_true',default=False,
                        help="Generate prediction of the  intermidate layer and last layer of the encoder"
                             " instead of training."
                             " You must supply a pre-trained model and define all parameters to be the same"
                             " as in training. ")
    parser.add_argument("-d","--dropout",action='store_true',default=False,
                        help="Use dropout")
    parser.add_argument("-l","--qlinear",action='store_true',default=False,
                        help="Perform a deterministic linear transformation instead of sampling"
                             " on the intermidiate layer")
    parser.add_argument("-s","--sparse",type=int,
                        help="Use sparse weight initialization. Give the number of non zero weights")
    args = parser.parse_args()
    main(**vars(args))
