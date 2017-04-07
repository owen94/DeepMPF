'''
This is a deep belief network trained with stacked RBM.
'''

from utils_mpf import *
from dmpf_optimizer import dmpf_optimizer
import timeit
from PIL import Image
from logistic_sgd import *
plt.switch_backend('agg')

class dbn(object):

    def __init__(self, n_ins=784,hidden_layers_sizes=[500, 500], n_outs=10, batch_sz = 40):



        self.n_ins = n_ins
        self.hidden_list = hidden_layers_sizes

        self.n_rbms = len(hidden_layers_sizes)
        self.dmpf_layers = []
         # the data is presented as rasterized images
        self.y = T.ivector('y')
        self.x = []

        for i in range(self.n_rbms):

            self.x  +=  [T.matrix('x_' + str(i))]
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            RBM = dmpf_optimizer(
                visible_units = input_size,
                hidden_units = hidden_layers_sizes[i],
                W = None,
                b = None,
                input = self.x[i],
                explicit_EM= True,
                batch_sz = 40)

            self.dmpf_layers.append(RBM)

        self.batch_sz = batch_sz



    def train_a_rbm(self, rbm, i, lr = 0.001, decay = 0.0001, sparsity = 0.1, beta= 0.01, sparsity_decay = 0.9,
                    dataset = None, epoches = 300):

        epoches = epoches
        in_epoch = 1
        path = '../deep/rbm_' + str(rbm.visible_units) + '_' + str(rbm.hidden_units) + '/lr_' + str(lr) + \
               '/decay_' + str(decay) + '/sparsity_' + str(sparsity) +  '/beta_' + str(beta)

        if not os.path.exists(path):
            os.makedirs(path)

        save_hidden = path + '/hidden_act.npy'

        if os.path.exists(save_hidden):
            print('This RBM is already trained before.')
            return save_hidden

        else:
            index = T.lscalar('index')

            if dataset is None:
                #data = load_mnist()
                data = load_IMAGE()
            else:
                data = np.load(dataset) ##  A rbm should return the activations in the hidden layer as the
                # input to the next rbm. Here we use .npy to save it.
                print(data.shape)

            visible_units = data.shape[1]
            num_samples = data.shape[0]

            num_batches = num_samples // self.batch_sz

            cost,updates = rbm.get_dmpf_cost(
                learning_rate= lr,
                decay= decay,
                sparsity= sparsity,
                beta= beta,
                sparse_decay= sparsity_decay)

            num_units = visible_units + rbm.hidden_units

            new_data  = theano.shared(value=np.asarray(np.zeros((data.shape[0],num_units)), dtype=theano.config.floatX),
                                      name = 'train',borrow = True)

            train_mpf = theano.function(
                [index],
                cost,
                updates=updates,
                givens={
                 self.x[i]: new_data[index * self.batch_sz: (index + 1) * self.batch_sz],
                    },
            #on_unused_input='warn',
                )

            #############  Start Training #########################
            mean_epoch_error = []

            start_time = timeit.default_timer()

            for em_epoch in range(epoches):

                if dataset is None:
                    binary_data = data
                else:
                    binary_data = np.random.binomial(n=1, p = data)

                W = rbm.W.get_value(borrow = True)
                b = rbm.b.get_value(borrow = True)

                prop_W = W[:visible_units, visible_units:]
                prop_b = b[visible_units:]
                activations, sample_data = get_new_data(binary_data,prop_W,prop_b)
                new_data.set_value(np.asarray(sample_data, dtype=theano.config.floatX))

                for mpf_epoch in range(in_epoch):
                    mean_cost = []
                    for batch_index in range(num_batches):
                        mean_cost += [train_mpf(batch_index)]
                    mean_epoch_error += [np.mean(mean_cost)]
                print('The cost for mpf in epoch %d is %f'% (em_epoch,mean_epoch_error[-1]))


                if int(em_epoch+1) % 50 ==0:

                    saveName = path + '/weights_' + str(em_epoch) + '.png'
                    tile_shape = (int(np.sqrt(rbm.hidden_units)), int(np.sqrt(rbm.hidden_units)))
                    image_shape = (int(np.sqrt(visible_units)), int(np.sqrt(visible_units)))

                    #displayNetwork(W1.T,saveName=saveName)

                    image = Image.fromarray(
                        tile_raster_images(  X=(rbm.W.get_value(borrow = True)[:visible_units,visible_units:]).T,
                                img_shape=image_shape,
                                tile_shape=tile_shape,
                                tile_spacing=(1, 1)
                            )
                            )
                    image.save(saveName)
                    W = rbm.W.get_value(borrow = True)
                    W1 = W[:visible_units,visible_units:]
                    b1 = rbm.b.get_value(borrow = True)

                    saveName_w = path + '/weights_' + str(em_epoch) + '.npy'
                    saveName_b = path + '/bias_' + str(em_epoch) + '.npy'
                    np.save(saveName_w,W1)
                    np.save(saveName_b,b1)

            hidden_activations = rbm.np_propup(vis=data)
            np.save(save_hidden, hidden_activations)

            return save_hidden

    def pretrain(self,lr = 0.001, decay = 0.0001, sparsity = 0.1, beta= 0.01, sparsity_decay = 0.9, epoches = 300 ):

        i = 0
        save_hidden = None

        for rbm in self.dmpf_layers:

            print('Pretraining the %d th RBM' % (i+1) )

            if i == 0:
                save_hidden = self.train_a_rbm(rbm=rbm, i = i, lr = lr, decay = decay, sparsity = sparsity,
                                               beta= beta, sparsity_decay = sparsity_decay,epoches=epoches)
            else:
                save_hidden = self.train_a_rbm(rbm=rbm,i = i, lr = lr, decay = decay, sparsity = sparsity,
                                               beta= beta, sparsity_decay = sparsity_decay, dataset=save_hidden,
                                               epoches= epoches)
            i += 1
            print(save_hidden)

    def build_classifier(self):

        ######  This function implement a classifier to fintune the whole model ###########

        ##  Construct a deep belief network with: Hidden Layer, logistic layer, and then optimize with a
        #   finetune cost #  ######

        self.sigmoid_layers = []
        numpy_rng = numpy.random.RandomState(1234)
        self.z = T.matrix('z')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector
                                 # of [int] label
        self.params = []
        for i in range(self.n_rbms):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = self.n_ins
            else:
                input_size = self.hidden_list[i-1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.z
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=self.hidden_list[i],
                                        W = self.dmpf_layers[i].W.get_value(borrow = True)[:input_size,input_size:],
                                        b = self.dmpf_layers[i].b.get_value(borrow = True)[input_size:],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=self.hidden_list[-1],
            n_out= 10 )
        self.params.extend(self.logLayer.params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def fine_tuning(self, datasets, batch_size, learning_rate):


        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = n_valid_batches // batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches = n_test_batches // batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.z: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.z: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.z: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score



def train_deep_rbm(lr, decay, sparsity, beta, sparsity_decay, hidden_list, epoches= 300, batch_size = 40 ):

    epoches = epoches
    lr = lr
    decay = decay
    sparsity = sparsity
    beta= beta
    sparsity_decay = sparsity_decay
    batch_size = batch_size
    hidden_list = hidden_list


    deep_belief_network = dbn(n_ins=64, hidden_layers_sizes=hidden_list,n_outs=10,batch_sz=40)

    deep_belief_network.pretrain(lr=lr,decay=decay,sparsity=sparsity,beta=beta,sparsity_decay=sparsity_decay, epoches=epoches)

    deep_belief_network.build_classifier()

    dataset = load_data('mnist.pkl.gz')
    train_fn, valid_model, test_model = deep_belief_network.fine_tuning(datasets= dataset,
                                                                        batch_size= batch_size, learning_rate=0.05)
    n_train_batches = dataset[0][0].get_value(borrow=True).shape[0] // batch_size



    mean_epoch_error = []
    test_epoch_error = []

    for epoch in range(epoches):
        mean_batch_error = []
        for batch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(batch_index)
            mean_batch_error += [minibatch_avg_cost]

        mean_epoch_error  += [np.mean(mean_batch_error)]

        test_loss = np.mean(test_model())

        test_epoch_error += [test_loss]
        print('The classification error for epoch %d is %f. ' % (epoch, test_epoch_error[-1]))

    path = '../deep/rbm_' + str(hidden_list[0]) + '_' + str(hidden_list[1]) + '/lr_' + str(lr) + \
               '/decay_' + str(decay) + '/sparsity_' + str(sparsity) +  '/beta_' + str(beta)
    show_loss(savename=path+'/train_error.png',epoch_error=mean_epoch_error)
    show_loss(savename=path + '/test_error.png', epoch_error = test_epoch_error)

    save(filename= path + '/rbm.pkl', bob=deep_belief_network)



if __name__ == '__main__':

    train_deep_rbm(lr=0.001,decay=0.0001,hidden_list=[196],
                                      beta=0,sparsity=0.1,sparsity_decay=0, epoches=300)

    # lr_list = [0.001, 0.0001]
    # decay_list = [0.0001, 0.001, 0.00001]
    # sparsity_list = [0.1, 0.2, 0.05]
    # beta_list = [0, 0.01, 0.1]
    # sparsity_decay_list = [0.9]
    # hidden_list_list = [[196, 100], [196, 64]]
    # epoches = 300
    #
    # for sparsity_decay in sparsity_decay_list:
    #     for lr in lr_list:
    #         for hidden_list in hidden_list_list:
    #             for decay in decay_list:
    #                 for beta in beta_list:
    #                     for sparsity in sparsity_list:
    #                         train_deep_rbm(lr=lr,decay=decay,hidden_list=hidden_list,
    #                                        beta=beta,sparsity=sparsity,sparsity_decay=sparsity_decay, epoches=epoches)


