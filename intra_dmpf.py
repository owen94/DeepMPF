'''
In this file we implement mpf with connection within layers.
Two things we need to do is:
1. change the initialization of weights and the update rule as well. (hidden connections are no longer zero)
2. Do the gibbs sampling to generate hidden states

'''

from dmpf_optimizer import *
from sklearn import preprocessing
import timeit, pickle, sys, math
from PIL import Image
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from utils_mpf import *
from comp_likelihood import *



FEED_FIRST = True







def intra_dmpf(hidden_units,learning_rate, epsilon, epoch = 200,  decay =0.0001,  batch_sz = 40, dataset = None,
           n_round = 1):

    ################################################################
    ################## Loading the Data        #####################
    ################################################################

    if dataset is None:
        dataset = 'mnist.pkl.gz'
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
        f.close()

    else:
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = pickle.load(dataset,encoding="bytes")
        f.close()


    binarizer = preprocessing.Binarizer(threshold=0.5)
    data =  binarizer.transform(train_set[0])
    print(data.shape)

    path = '../intra_mpf/hidden_' + str(hidden_units) + '/decay_' + str(decay) + '/lr_' + str(learning_rate)

    if not os.path.exists(path):
        os.makedirs(path)

    visible_units = data.shape[1]

    n_train_batches = data.shape[0]//batch_sz

    num_units = visible_units + hidden_units

    W = get_intra_mpf_params(visible_units,hidden_units)
    b = np.zeros(num_units)

    index = T.lscalar()    # index to a mini batch
    x = T.matrix('x')

    mpf_optimizer = dmpf_optimizer(
        epsilon=epsilon,
        explicit_EM= explicit_EM,
        hidden_units= hidden_units,
        W = W,
        b = b,
        input = x,
        batch_sz =batch_sz)


    new_data = theano.shared(value=np.asarray(data,dtype=theano.config.floatX),name= 'mnist',borrow = True)


    cost,updates = mpf_optimizer.intra_dmpf_cost(n_round= n_round,
        learning_rate= learning_rate, decay=decay, feed_first= FEED_FIRST)

    train_mpf = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
        x: new_data[index * batch_sz: (index + 1) * batch_sz],
        },
        #on_unused_input='warn',
    )


    mean_epoch_error = []

    test_data = test_set[0]
    train_data = train_set[0]

    test_lld = []
    test_std = []

    start_time = timeit.default_timer()

    for epoch_i in range(epoch):
        mean_cost = []
        for batch_index in range(n_train_batches):
            mean_cost += [train_mpf(batch_index)]
        mean_epoch_error += [np.mean(mean_cost)]
        print('The cost for mpf in epoch %d is %f'% (epoch_i,mean_epoch_error[-1]))


        if int(epoch_i+1) % 100 ==0:

                saveName = path + '/weights_' + str(epoch_i) + '.png'
                tile_shape = (10, hidden_units//10)

                #displayNetwork(W1.T,saveName=saveName)

                image = Image.fromarray(
                    tile_raster_images(  X=(mpf_optimizer.W.get_value(borrow = True)[:visible_units,visible_units:]).T,
                            img_shape=(28, 28),
                            tile_shape=tile_shape,
                            tile_spacing=(1, 1)
                        )
                        )
                image.save(saveName)
                W = mpf_optimizer.W.get_value(borrow = True)
                W1 = W[:visible_units,visible_units:]
                b1 = mpf_optimizer.b.get_value(borrow = True)

                saveName_w = path + '/weights_' + str(epoch_i) + '.npy'
                saveName_b = path + '/bias_' + str(epoch_i) + '.npy'
                np.save(saveName_w,W1)
                np.save(saveName_b,b1)



if __name__ == '__main__':


    learning_rate_list = [0.001]
    # hyper-parameters are: learning rate, num_samples, sparsity, beta, epsilon, batch_sz, epoches
    # Important ones: num_samples, learning_rate,
    hidden_units_list = [196]
    n_samples_list = [1]
    beta_list = [0]
    sparsity_list = [0]
    batch_list = [40]
    decay_list = [0.0001]

    for batch_size in batch_list:
        for n_samples in n_samples_list:
            for hidden_units in hidden_units_list:
                for decay in decay_list:
                    for learning_rate in learning_rate_list:
                            savename_w, savename_b = intra_dmpf(hidden_units = hidden_units,learning_rate = learning_rate, epsilon = 0.01,decay=decay,
                                   batch_sz=batch_size)