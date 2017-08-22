'''
...This file provide a training procedure regarding the Deep boltzmann machine.
DBM differenciate from DBN since it is undirected graphical model.
'''
from utils_mpf import *
from theano.tensor.shared_randomstreams import RandomStreams
from theano_optimizers import Adam
import os
import timeit
from PIL import Image
from comp_likelihood import gpu_parzen, get_ll




class DBM(object):

    def __init__(self, hidden_list = [] , batch_sz = 40, input1 = None, input2 = None, input3 = None):

        self.num_rbm = int(len(hidden_list) - 1 )
        self.hidden_list = hidden_list

        self.W = []
        self.b = []

        for i in range(self.num_rbm):
            initial_W = np.asarray(get_mpf_params(self.hidden_list[i],self.hidden_list[i+1]),
                dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

            self.W.append(W)

            num_neuron = hidden_list[i] + hidden_list[i+1]
            b = theano.shared(value=np.zeros(num_neuron,dtype=theano.config.floatX),
                name='bias',borrow=True)
            self.b.append(b)

        numpy_rng = np.random.RandomState(1233456)

        self.theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.epsilon = 0.01

        self.batch_sz = batch_sz

        self.input1 = input1
        self.input2 = input2
        self.input3 = input3

        if len(hidden_list) == 4:
            self.input = [self.input1, self.input2, self.input3]
        elif len(hidden_list) == 3:
            self.input = [self.input1, self.input2]



    def get_cost_update(self,decay_list = [], learning_rate = 0.001):
        '''
        :param undirected_act: The input from the forward and backward pass
        :return: the cost function and the updates
        '''
        updates = []
        cost = 0
        for i in range(self.num_rbm):

            decay = decay_list[i]

            W = self.W[i]
            b = self.b[i]

            z = 1/2 - self.input[i]
            energy_difference = z * (T.dot(self.input[i], W)+ b.reshape([1,-1]))

            cost = (self.epsilon/self.batch_sz) * T.sum(T.exp(energy_difference))
            cost_weight = 0.5 * decay * T.sum(W**2)
            cost += cost_weight

            h = z * T.exp(energy_difference)
            W_grad = (T.dot(h.T,self.input[i])+T.dot(self.input[i].T,h))*self.epsilon/self.batch_sz
            b_grad = T.mean(h,axis=0)*self.epsilon
            decay_grad = decay*W
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

            params = [self.W[i],self.b[i]]

            update_rbm = Adam(grads=grads, params=params,lr=learning_rate)
            updates += update_rbm

        return cost, updates


def train_dbm(hidden_list, decay, lr, undirected = False,  batch_sz = 40, epoch = 400):

    data = load_mnist()

    num_rbm = len(hidden_list) -1
    index = T.lscalar()    # index to a mini batch
    x1 = T.matrix('x1')
    x2 = T.matrix('x2')
    x3 = T.matrix('x3')

    if len(hidden_list) == 4:

        if undirected:
            path = '../DBM_results/Undirected_DBM/DBM_' + str(hidden_list[1]) + '_' + str(hidden_list[2]) \
               + '_' + str(hidden_list[3]) + '/decay_' + str(decay[1]) + '/lr_' + str(lr)
        else:
            path = '../DBM_results/Directed_DBM/DBM_' + str(hidden_list[1]) + '_' + str(hidden_list[2]) \
               + '_' + str(hidden_list[3]) + '/decay_' + str(decay[1]) + '/lr_' + str(lr)
        dbm = DBM(hidden_list = hidden_list,
              input1=x1,
              input2=x2,
              input3=x3,
              batch_sz=batch_sz)

    elif len(hidden_list) ==3:
        if undirected:
            path = '../DBM_results/Undirected_DBM/DBM_' + str(hidden_list[1]) + '_' + str(hidden_list[2]) \
               + '/decay_' + str(decay[1]) + '/lr_' + str(lr)
        else:
            path = '../DBM_results/Directed_DBM/DBM_' + str(hidden_list[1]) + '_' + str(hidden_list[2]) \
               + '/decay_' + str(decay[1]) + '/lr_' + str(lr)

        dbm = DBM(hidden_list = hidden_list,
              input1=x1,
              input2=x2,
              batch_sz=batch_sz)

    if not os.path.exists(path):
        os.makedirs(path)

    n_train_batches = data.shape[0]//batch_sz

    new_data = []

    for i in range(num_rbm):
        num_units = hidden_list[i] + hidden_list[i+1]
        new_data.append(theano.shared(value=np.asarray(np.zeros((data.shape[0],num_units)), dtype=theano.config.floatX),
                                  name = 'train',borrow = True) )

    cost, updates = dbm.get_cost_update(decay_list=decay,learning_rate=lr)

    if len(hidden_list) == 4:
        train_func = theano.function([index], cost, updates= updates,
                                 givens= {
                                     x1: new_data[0][index * batch_sz: (index + 1) * batch_sz],
                                     x2: new_data[1][index * batch_sz: (index + 1) * batch_sz],
                                     x3: new_data[2][index * batch_sz: (index + 1) * batch_sz]
                                 })
    elif len(hidden_list) ==3:
        train_func = theano.function([index], cost, updates= updates,
                                 givens= {
                                     x1: new_data[0][index * batch_sz: (index + 1) * batch_sz],
                                     x2: new_data[1][index * batch_sz: (index + 1) * batch_sz],
                                 })

    mean_epoch_error = []
    start_time = timeit.default_timer()

    train_lld = []
    train_std = []
    test_lld = []
    test_std = []

    for n_epoch in range(epoch):

        ## propup to get the trainning data

        W = []
        b = []
        for i in range(num_rbm):
            W.append(dbm.W[i].get_value(borrow = True))
            b.append(dbm.b[i].get_value(borrow = True))

        samplor = get_samples(hidden_list= hidden_list, W=W, b = b)
        forward_act, forward_data = samplor.forward_pass(input_data= data)

        if undirected:
            undirected_data = samplor.undirected_pass(forward_act = forward_act)
            for j in range(num_rbm):
                new_data[j].set_value(np.asarray(undirected_data[j], dtype=theano.config.floatX))
        if not undirected:
            for j in range(num_rbm):
                new_data[j].set_value(np.asarray(forward_data[j], dtype=theano.config.floatX))


        ## Train the dbm
        mean_cost = []
        for batch_index in range(n_train_batches):
            mean_cost += [train_func(batch_index)]
        mean_epoch_error += [np.mean(mean_cost)]
        print('The cost for mpf in epoch %d is %f'% (n_epoch,mean_epoch_error[-1]))


        # if int(n_epoch+1) % 20 ==0:
        #
        #     saveName = path + '/weights_' + str(n_epoch) + '.png'
        #     tile_shape = (10, hidden_list[1]//10)
        #
        #     #displayNetwork(W1.T,saveName=saveName)
        #
        #     W = dbm.W[0].get_value(borrow = True)
        #     visible_units = hidden_list[0]
        #
        #     image = Image.fromarray(
        #         tile_raster_images(  X=(W[:visible_units,visible_units:]).T,
        #                 img_shape=(28, 28),
        #                 tile_shape=tile_shape,
        #                 tile_spacing=(1, 1)
        #             )
        #             )
        #     image.save(saveName)

        if int(n_epoch+1) % 100 ==0:
            filename = path + '/dbm_' + str(n_epoch) + '.pkl'
            save(filename,dbm)

            W = []
            b = []
            for i in range(num_rbm):
                W.append(dbm.W[i].get_value(borrow = True))
                b.append(dbm.b[i].get_value(borrow = True))

            w_name = path + '/weight_' + str(n_epoch) + '.npy'
            b_name = path + '/bias_' + str(n_epoch) + '.npy'
            np.save(w_name,W)
            np.save(b_name,b)

            n_chains = 20
            n_samples = 10
            plot_every = 7
            image_data = np.zeros(
                (29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8'
            )

            for idx in range(n_samples):
                persistent_vis_chain = np.random.randint(2,size=(n_chains, hidden_list[-1]))

                v_samples = persistent_vis_chain

                for i in range(num_rbm):

                    vis_units = hidden_list[num_rbm-i - 1]
                    W_sample = W[num_rbm - i -1 ][:vis_units,vis_units:]
                    b_down = b[num_rbm - i -1 ][:vis_units]
                    b_up = b[num_rbm - i -1 ][vis_units:]

                    for j in range(plot_every):
                        downact1 = sigmoid(np.dot(v_samples,W_sample.T) + b_down )
                        down_sample1 = np.random.binomial(n=1, p= downact1)
                        upact1 = sigmoid(np.dot(down_sample1,W_sample)+b_up)
                        v_samples = np.random.binomial(n=1,p=upact1)
                    v_samples = down_sample1
                print(' ... plotting sample ', idx)

                image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
                    X= downact1,
                    img_shape=(28, 28),
                    tile_shape=(1, n_chains),
                    tile_spacing=(1, 1)
                )

            image = Image.fromarray(image_data)
            image.save(path + '/samples_' + str(n_epoch) + '.png')

        if n_epoch % 5 == 0:
            W = []
            b = []
            for i in range(num_rbm):
                W.append(dbm.W[i].get_value(borrow = True))
                b.append(dbm.b[i].get_value(borrow = True))
            dataset = 'mnist.pkl.gz'
            f = gzip.open(dataset, 'rb')
            train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
            f.close()

            binarizer = preprocessing.Binarizer(threshold=0.5)
            training_data =  binarizer.transform(train_set[0])
            test_data = test_set[0]
            train_data = train_set[0]

            ##############################################################################
            n_sample = 10000
            plot_every = 5
            ################################################################################
            # for i in range(num_rbm):
            #     feed_vis_units = hidden_list[i]
            #     feed_w = W[i][:feed_vis_units,feed_vis_units:]
            #     feed_b = b[i][feed_vis_units:]
            #     feed_data = sigmoid(np.dot(feed_data, feed_w) + feed_b)
            error_bar_lld = []
            error_bar_std = []

            #for kk in range(1):

            feed_samplor = get_samples(hidden_list=hidden_list, W=W, b=b)
            feed_data = feed_samplor.get_mean_activation(input_data= training_data)

            feed_mean_activation = np.mean(feed_data, axis=0)
            feed_initial = np.random.binomial(n=1, p= feed_mean_activation, size=(n_sample, hidden_list[-1]))
            ###########################################################

            ######### generate the parzen sample to compute the model distribution ###########
            v_samples = feed_initial
            for i in range(num_rbm):
                vis_units = hidden_list[num_rbm-i - 1]
                W_sample = W[num_rbm - i -1 ][:vis_units,vis_units:]
                b_down = b[num_rbm - i -1 ][:vis_units]
                b_up = b[num_rbm - i -1 ][vis_units:]

                for j in range(plot_every):
                    downact1 = sigmoid(np.dot(v_samples,W_sample.T) + b_down )
                    down_sample1 = np.random.binomial(n=1, p= downact1)
                    upact1 = sigmoid(np.dot(down_sample1,W_sample)+b_up)
                    v_samples = np.random.binomial(n=1,p=upact1)

                v_samples = down_sample1

            parzen_sample = downact1
            # compute the log-likelihood for the training data
            epoch_train_lld = get_ll(x=train_data[:10000],
                                     gpu_parzen=gpu_parzen(mu=parzen_sample,sigma=0.2),batch_size=20)
            train_mean_lld = np.mean(np.array(epoch_train_lld))
            train_std_lld = np.std(np.array(epoch_train_lld))
            train_lld += [train_mean_lld]
            train_std += [train_std_lld]


            # comppute the log-likelihood for the test data
            epoch_test_lld = get_ll(x=test_data, gpu_parzen=gpu_parzen(mu=parzen_sample,sigma=0.2),batch_size=10)
            test_mean_lld = np.mean(np.array(epoch_test_lld))
            test_std_lld = np.std(np.array(epoch_test_lld))
            test_lld += [test_mean_lld]
            test_std += [test_std_lld]

            print('The loglikehood in epoch {} is: train {}, test {}'.format(n_epoch, train_mean_lld, test_mean_lld))

    path_1 = path + '/train_lld.npy'
    path_2 = path + '/train_std.npy'
    path_3 = path + '/test_lld.npy'
    path_4 = path + '/test_std.npy'


    np.save(path_1, train_lld)
    np.save(path_2, train_std)
    np.save(path_3, test_lld)
    np.save(path_4, test_std)

    print('...............................................')
    print(train_lld)
    print('...............................................')
    print(test_lld)

    loss_savename = path + '/train_loss.eps'
    show_loss(savename= loss_savename, epoch_error= mean_epoch_error)

    end_time = timeit.default_timer()

    running_time = (end_time - start_time)

    print ('Training took %f minutes' % (running_time / 60.))


    ###  generate samples ##########################

if __name__ == '__main__':


    learning_rate_list = [0.0001]
    # hyper-parameters are: learning rate, num_samples, sparsity, beta, epsilon, batch_sz, epoches
    # Important ones: num_samples, learning_rate,
    hidden_units_list = [[784, 196, 196, 64]]
    n_samples_list = [1]
    beta_list = [0]
    sparsity_list = [0]
    batch_list = [40]
    decay_list = [[0.0001, 0.01, 0.01, 0.01]]

    undirected_list = [False]
    for undirected in undirected_list:
        for learning_rate in learning_rate_list:
            for hidden_list in hidden_units_list:
                for decay in decay_list:
                    train_dbm(hidden_list=hidden_list,decay=decay,lr=learning_rate, undirected=undirected)


