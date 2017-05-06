'''
This file provide a training procedure regarding the Deep boltzmann machine.
DBM differenciate from DBN since it is undirected graphical model.
'''
from utils_mpf import *
from theano.tensor.shared_randomstreams import RandomStreams
from theano_optimizers import Adam


class DBM(object):

    def __init__(self, hidden_list = [] , batch_sz = 40, input = None):

        self.num_rbm = int(len(hidden_list) - 1 )
        self.hidden_list = hidden_list

        self.W = []
        self.b = []
        #self.x = []

        for i in range(self.num_rbm):
            initial_W = np.asarray(get_mpf_params(self.hidden_list[i],self.hidden_list[i+1]),
                dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

            self.W.append(W)

            num_neuron = hidden_list[i] + hidden_list[i+1]
            b = theano.shared(value=np.zeros(num_neuron,dtype=theano.config.floatX),
                name='bias',borrow=True)
            self.b.append(b)

            #self.x  +=  [T.matrix('x_' + str(i))]

        numpy_rng = np.random.RandomState(1233456)

        self.theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.epsilon = 0.01

        self.batch_sz = batch_sz

        if not input:
            self.input = T.matrix('input')
        else:
            self.input = input

    def propup(self, i, data):
        vis_units = self.hidden_list[i]
        pre_sigmoid_activation = T.dot(data, self.W[i][:vis_units,vis_units:]) \
                                 + self.b[i][vis_units:]
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def forward_pass(self, input_data):
        '''
        input: the input dataset
        :return: a binary dataset with hidden states can be used for vpf
        '''
        forward_act = []
        forward_act.append(input_data)
        for i in range(self.num_rbm):
            act = self.propup(i, data = forward_act[i])
            act_sample = self.theano_rng.binomial(size=act[1].shape,
                                             n=1, p=act,
                                             dtype=theano.config.floatX)
            forward_act.append(act_sample)
        return forward_act

    def undirected_pass(self,forward_act):

        num_intermediate = self.num_rbm - 1
        undirected_act = []
        undirected_act.append(forward_act[0])
        for i in range(num_intermediate):
            bottom_W = self.W[i]
            bottom_b = self.b[i]
            up_W = self.W[i+1]
            up_b = self.b[i+1]
            bottom_vis = self.hidden_list[i]
            up_vis = self.hidden_list[i+2]

            pre_inter_act = T.dot(forward_act[i],bottom_W[:bottom_vis,bottom_vis:] + bottom_b[bottom_vis:]) + \
                            T.dot(forward_act[i+2],up_W[:bottom_vis,bottom_vis:].T + up_b[:up_vis])
            inter_act = T.nnet.sigmoid(pre_inter_act)
            inter_act_sample = self.theano_rng.binomial(size=inter_act.shape,
                                             n=1, p=inter_act,
                                             dtype=theano.config.floatX)
            undirected_act.append(inter_act_sample)

        undirected_act.append(forward_act[-1])


        undirected_data = []
        for i in range(self.num_rbm):
            data = T.concatenate((undirected_act[i], undirected_act[i+1]), axis = 0)
            undirected_data.append(data)
        return undirected_data


    def get_cost_update(self,undirected_act,decay = 0.00001, learning_rate = 0.001):
        '''
        :param undirected_act: The input from the forward and backward pass
        :return: the cost function and the updates
        '''
        updates = []
        cost = 0
        decay = decay
        for i in range(self.num_rbm):

            input = T.concatenate((undirected_act[i],undirected_act[i+1]), axis = 0)
            W = self.W[i]
            b = self.b[i]

            z = 1/2 - input
            energy_difference = z * (T.dot(input, W)+ b.reshape([1,-1]))

            cost = (self.epsilon/self.batch_sz) * T.sum(T.exp(energy_difference))
            cost_weight = 0.5 * decay * T.sum(W**2)
            cost += cost_weight

            h = z * T.exp(energy_difference)
            W_grad = (T.dot(h.T,input)+T.dot(input.T,h))*self.epsilon/self.batch_sz
            b_grad = T.mean(h,axis=0)*self.epsilon
            decay_grad = decay*self.W
            W_grad += decay_grad

            visible_units = self.hidden_list[i]
            hidden_units = self.hidden_list[i+1]
            a = np.ones((visible_units,hidden_units))
            b = np.zeros((visible_units,visible_units))
            c = np.zeros((hidden_units,hidden_units))
            zero_grad_u = np.concatenate((b,a),axis = 1)
            zero_grad_d = np.concatenate((a.T,c),axis=1)
            zero_grad = np.concatenate((zero_grad_u,zero_grad_d),axis=0)
            zero_grad = theano.shared(value=np.asarray(zero_grad,dtype=theano.config.floatX),
                                           name='zero_grad',borrow = True)
            W_grad *= zero_grad
            grads = [W_grad,b_grad]

            params = [W,b]

            update_rbm = Adam(grads=grads, params=params,lr=learning_rate)
            updates += update_rbm

        return cost, updates






def train_dbm(hidden_list, decay, lr):

    data = load_mnist()

    '''
    step 1: forward pass
    step 2: undirected pass to generate samples
    step 3: given undirected data, do sgd
    step 4: go to step 1 again after
    '''


    dbm = dbm(hidden_list = hidden_list,)