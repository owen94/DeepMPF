'''
This optimizer is developed for solving the MPF objective function with binary inputs.
We will view the entire graph which needs to be finetuned as a fully-observable boltzmann machine.
The traintime could benefit a lot from the special form of MPF.
'''

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import sys
sys.setrecursionlimit(40000)
from theano_optimizers import Adam
from utils_mpf import *


'''Optimizes based on MPF for fully-observable boltzmann machine'''
class dmpf_optimizer(object):

    def __init__(self, visible_units = 784, hidden_units = 196, W = None, b = None,
                 input = None, batch_sz = 20,
                 explicit_EM = True,theano_rng = None, epsilon = 0.01 ):
        '''
        :param visible_units:
        :param hidden_units:
        :param W:
        :param b:
        :param input:
        :param decay: The weight decay regularization term
        :param batch_sz:
        :param explicit_EM:
        :param theano_rng:
        :param epsilon:
        :return:
        '''
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        num_units = visible_units + hidden_units
        self.num_neuron = num_units

        numpy_rng = np.random.RandomState(1233456)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

        if W is None:
            initial_W = np.asarray(get_mpf_params(visible_units, hidden_units),
                dtype=theano.config.floatX)
            self.W = theano.shared(value=initial_W, name='W', borrow=True)
        else:
            self.W = theano.shared(value=np.asarray(W,dtype=theano.config.floatX),name = 'Weight', borrow = True)

        if b is None:
            self.b = theano.shared(value=np.zeros(self.num_neuron,dtype=theano.config.floatX),
                name='bias',borrow=True)
        else:
            self.b = theano.shared(value=np.asarray(b,dtype=theano.config.floatX),name = 'bias', borrow = True)

        self.epsilon = epsilon

        self.batch_sz = batch_sz


        if not input:
            self.input = T.matrix('input')
        else:
            self.input = input

        self.params = [self.W, self.b]

        self.zero_grad = None

        if self.zero_grad is None:
            a = np.ones((visible_units,hidden_units))
            b = np.zeros((visible_units,visible_units))
            c = np.zeros((hidden_units,hidden_units))
            zero_grad_u = np.concatenate((b,a),axis = 1)
            zero_grad_d = np.concatenate((a.T,c),axis=1)
            zero_grad = np.concatenate((zero_grad_u,zero_grad_d),axis=0)
            self.zero_grad = theano.shared(value=np.asarray(zero_grad,dtype=theano.config.floatX),
                                           name='zero_grad',borrow = True)

        self.explicit_EM = explicit_EM

        self.rho = 0

        self.intra_grad = None
        if self.intra_grad is None:
            a = np.ones((visible_units,hidden_units))
            b = np.zeros((visible_units,visible_units))
            c = np.ones((hidden_units,hidden_units)) - np.diagflat(np.ones(hidden_units))
            assert c[0,0] == 0

            intra_grad_u = np.concatenate((b,a), axis = 1)
            intra_grad_d = np.concatenate((a.T, c), axis=1)
            intra_grad = np.concatenate((intra_grad_u,intra_grad_d), axis = 0)
            self.intra_grad = theano.shared(value=np.asarray(intra_grad,dtype=theano.config.floatX),
                                            name='intra_grad', borrow = True)


        #self.params = []

    def get_dmpf_cost(self, learning_rate = 0.001, decay=0.0001, beta=0, sparsity = 0.2, sparse_decay = 0.9):

        # In one round, we feed forward the and get the samples,
        # Compute the probability of each data samples,
        # call the minimum probability flow objective function
        if not self.explicit_EM:

            ####################### one sample MPF ##############################
            ##############################################################
            activation = self.sample_h_given_v(v0_sample=self.input)
            hidden_samples = activation[2]
            self.input = T.concatenate((self.input, hidden_samples),axis = 1)

        z = 1/2 - self.input
        energy_difference = z * (T.dot(self.input,self.W)+ self.b.reshape([1,-1]))

        # self.sample_prob = self.sample_prob.reshape((1,-1)).T
        # k = theano.shared(value= (np.asarray(np.ones(self.num_neuron),dtype=theano.config.floatX)).reshape((1,-1)))
        # self.sample_prob = T.dot(self.sample_prob, k)
        cost = (self.epsilon/self.batch_sz) * T.sum(T.exp(energy_difference))
        cost_weight = 0.5 * decay * T.sum(self.W**2)
        cost += cost_weight

        h = z * T.exp(energy_difference)
        W_grad = (T.dot(h.T,self.input)+T.dot(self.input.T,h))*self.epsilon/self.batch_sz
        b_grad = T.mean(h,axis=0)*self.epsilon
        decay_grad = decay*self.W
        W_grad += decay_grad

        ###############   Add  sparsity Here ###########################

        if beta != 0:

            raw = self.input[:,:self.visible_units]
            activation = self.propup(raw)[1]
            rho = T.mean(activation,axis=0)  # rho is the current
            self.rho = rho * (1 - sparse_decay) + sparse_decay * self.rho

            cost_kl = beta*T.sum(sparsity* T.log(sparsity/self.rho)+(1 - sparsity)*T.log((1 - sparsity)/(1 - self.rho)))

            KL = beta*((-sparsity/self.rho) + ((1 - sparsity)/(1 - self.rho)))
            kl_b_grad = KL*activation*(1-activation)

            KL_grad = T.dot(raw.T, kl_b_grad)

            a = theano.shared(value = np.asarray(np.zeros((self.visible_units,self.visible_units)),dtype=theano.config.floatX)
                              , borrow = True)
            b = theano.shared(value = np.asarray(np.zeros((self.hidden_units,self.hidden_units)),dtype=theano.config.floatX)
                              , borrow = True)
            kl_grad1 = T.concatenate((a, KL_grad),axis=1)
            kl_grad2 = T.concatenate((KL_grad.T, b),axis=1)
            kl_w_grad = T.concatenate((kl_grad1,kl_grad2),axis=0)
            ##########################################################

            cost += cost_kl
            W_grad += (kl_w_grad/self.batch_sz)
            b_grad = T.inc_subtensor(b_grad[self.visible_units:], T.mean(kl_b_grad,axis=0))

        W_grad *= self.zero_grad
        grads = [W_grad,b_grad]

        updates = Adam(grads=grads, params=self.params, lr=learning_rate)

        return cost, updates

    def propup(self, vis):
        pre_sigmoid_activation = T.dot(vis, self.W[:self.visible_units,self.visible_units:]) \
                                 + self.b[self.visible_units:]
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def np_propup(self, vis):
        pre_sigmoid_activation = np.dot(vis, self.W.get_value(borrow = True)[:self.visible_units,self.visible_units:]) \
                                 + self.b.get_value(borrow = True)[self.visible_units:]
        return sigmoid(pre_sigmoid_activation)

    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        pre_sigmoid_activation = T.dot(hid, self.W[:self.visible_units,self.visible_units:].T) \
                                 + self.b[:self.visible_units]
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]


    def intra_dmpf_cost(self, n_round =1,learning_rate = 0.001, decay=0.0001, feed_first = True):

        # feed self.input only as the data samples
        self.asyc_gibbs(n_round=n_round, feedforward= feed_first) # update self.input

        z = 1/2 - self.input
        energy_difference = z * (T.dot(self.input,self.W)+ self.b.reshape([1,-1]))
        cost = (self.epsilon/self.batch_sz) * T.sum(T.exp(energy_difference))
        cost_weight = 0.5 * decay * T.sum(self.W**2)
        cost += cost_weight

        h = z * T.exp(energy_difference)
        W_grad = (T.dot(h.T,self.input)+T.dot(self.input.T,h))*self.epsilon/self.batch_sz
        b_grad = T.mean(h,axis=0)*self.epsilon
        decay_grad = decay*self.W
        W_grad += decay_grad
        # Need to change the self.zero_grad to another multiplier
        W_grad *= self.intra_grad
        grads = [W_grad,b_grad]

        updates = Adam(grads=grads, params=self.params, lr=learning_rate)

        return cost, updates

    def one_gibbs(self, node_i):
        input_w = self.W[:,node_i]
        input_b = self.b[node_i]
        activations = T.nnet.sigmoid( T.dot(self.input,input_w) +input_b)
        flip = self.theano_rng.binomial(size=activations.shape,n=1,p=activations,dtype=theano.config.floatX)
        update_input = T.set_subtensor(self.input[:,node_i],flip)

        return update_input

    def asyc_gibbs(self,n_round = 1, feedforward = False):

        if feedforward:
            activation = self.sample_h_given_v(v0_sample=self.input)
            hidden_samples = activation[2]
            self.input = T.concatenate((self.input, hidden_samples),axis = 1)
        else:
            self.rand_h = self.theano_rng.binomial(size=(self.batch_sz,self.hidden_units))
            self.input = T.concatenate( (self.input,self.rand_h), axis = 1)

        #assert self.input.shape == (self.batch_sz,self.hidden_units + self.visible_units)

        for i in range(n_round):

            for j in range(self.hidden_units):

                node_j = j + self.visible_units

                self.input = self.one_gibbs(node_i= node_j)














