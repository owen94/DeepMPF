import numpy as np
from utils_mpf import *


class get_samples(object):

    def __init__(self, hidden_list = [], W = None, b = None):
        self.hidden_list = hidden_list
        self.num_rbm = len(hidden_list) - 1
        self.W = W
        self.b = b

    def propup(self, i, data):
        vis_units = self.hidden_list[i]
        pre_sigmoid_activation = np.dot(data, self.W[i][:vis_units,vis_units:]) \
                             + self.b[i][vis_units:]
        return [pre_sigmoid_activation, sigmoid(pre_sigmoid_activation)]


    def get_mean_activation(self, input_data):

        act = None
        for i in range(self.num_rbm):

            act = self.propup(i, data = input_data)
            input_data = act[1]

        return act[1]


    def forward_pass(self, input_data):
        '''
        input: the input dataset
        :return: a binary dataset with hidden states can be used for vpf
        '''
        forward_act = []
        forward_act.append(input_data)
        for i in range(self.num_rbm):
            act = self.propup(i, data = forward_act[i])
            act_sample = np.random.binomial(size=act[1].shape,
                                             n=1, p=act[1])
            forward_act.append(act_sample)

        forward_data = []

        # concatenate the states, generate the fully-visible states for training of DBM
        for i in range(self.num_rbm):
            data = np.concatenate((forward_act[i], forward_act[i+1]), axis = 1)
            forward_data.append(data)
        return forward_act, forward_data

    def undirected_pass(self,forward_act):

        # Taking into account both the botton and up layer of the intermediate layers
        num_intermediate = self.num_rbm - 1
        undirected_act = []
        undirected_act.append(forward_act[0])
        for i in range(num_intermediate):
            bottom_W = self.W[i]
            bottom_b = self.b[i]
            up_W = self.W[i+1]
            up_b = self.b[i+1]
            bottom_vis = self.hidden_list[i]
            up_vis = self.hidden_list[i+1]

            pre_inter_act = np.dot(forward_act[i],bottom_W[:bottom_vis,bottom_vis:] + bottom_b[bottom_vis:]) + \
                            np.dot(forward_act[i+2],up_W[:up_vis,up_vis:].T + up_b[:up_vis])
            inter_act = sigmoid(pre_inter_act)
            inter_act_sample = np.random.binomial(size=inter_act.shape,
                                             n=1, p=inter_act)
            undirected_act.append(inter_act_sample)

        undirected_act.append(forward_act[-1])

        undirected_data = []

        # concatenate the states, generate the fully-visible states for training of DBM
        for i in range(self.num_rbm):
            data = np.concatenate((undirected_act[i], undirected_act[i+1]), axis = 1)
            undirected_data.append(data)
        return undirected_data