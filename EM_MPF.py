'''
The explicit version of EM-MPF.
'''


from dmpf_optimizer import *
from sklearn import preprocessing
import timeit, pickle, sys, math
from PIL import Image
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from utils_mpf import *


def em_mpf(hidden_units,learning_rate, epsilon, epoch = 100,  decay =0.001,  batch_sz = 20, dataset = None,
           explicit_EM= False):

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

    path = '../Thea_mpf/hidden_' + str(hidden_units) + '/decay_' + str(decay) + '/lr_' + str(learning_rate) \
           + '/bsz_' + str(batch_sz)
    if not os.path.exists(path):
        os.makedirs(path)
    #displayNetwork(data[:100,:])
    # Binarize the mnist data doesnot hurt much to the input data.
    # displayNetwork(train_set[0][:100,:])


    ################################################################
    ##################  Initialize Parameters  #####################
    ################################################################

    #visible_units = train_set[0].shape[1]
    visible_units = data.shape[1]

    n_train_batches = data.shape[0]//batch_sz

    num_units = visible_units + hidden_units

    W = get_mpf_params(visible_units, hidden_units)

    b = np.zeros(num_units)

    out_epoch = epoch
    in_epoch = 1

    index = T.lscalar()    # index to a mini batch
    x = T.matrix('x')

    mpf_optimizer = dmpf_optimizer(
        epsilon=epsilon,
        explicit_EM= explicit_EM,
        W = W,
        b = b,
        input = x,
        batch_sz =batch_sz)


    if explicit_EM:
        new_data  = theano.shared(value=np.asarray(np.zeros((data.shape[0],num_units)), dtype=theano.config.floatX),
                                  name = 'train',borrow = True)
    else:
        new_data = theano.shared(value=np.asarray(data,dtype=theano.config.floaxX),name= 'mnist',borrow = True)



    cost,updates = mpf_optimizer.get_dmpf_cost(
        learning_rate= learning_rate)

    train_mpf = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
        x: new_data[index * batch_sz: (index + 1) * batch_sz],
        },
        #on_unused_input='warn',
    )

    saveName_w = None
    saveName_b = None
    mean_epoch_error = []

    start_time = timeit.default_timer()

    for em_epoch in range(out_epoch):

        if explicit_EM:

            W = mpf_optimizer.W.get_value(borrow = True)
            b = mpf_optimizer.b.get_value(borrow = True)

            prop_W = W[:visible_units, visible_units:]
            prop_b = b[visible_units:]
            activations, sample_data = get_new_data(data,prop_W,prop_b)

            #sample_prob = get_sample_prob(activations) # This is a vector, each entry stands for the probability of
            #the respected sample
            #y = T.vector('y')
            #new_data.set_value(np.asarray(sample_data, dtype=theano.config.floatX))
            # sample_prob = theano.shared(value = np.asarray(sample_prob, dtype= theano.config.floatX),
            #                             name='prob',borrow = True)
            #new_data.set_value(value=np.asarray(sample_data, dtype=theano.config.floatX),borrow = True)
            new_data.set_value(np.asarray(sample_data, dtype=theano.config.floatX))

        for mpf_epoch in range(in_epoch):
            mean_cost = []
            for batch_index in range(n_train_batches):
                mean_cost += [train_mpf(batch_index)]
            mean_epoch_error += [np.mean(mean_cost)]
        print('The cost for mpf in epoch %d is %f'% (em_epoch,mean_epoch_error[-1]))


        if int(em_epoch+1) % 20 ==0:

            saveName = path + '/weights_' + str(em_epoch) + '.png'
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

            saveName_w = path + '/weights_' + str(em_epoch) + '.npy'
            saveName_b = path + '/bias_' + str(em_epoch) + '.npy'
            np.save(saveName_w,W1)
            np.save(saveName_b,b1)


        if em_epoch > 0 and em_epoch % 100 == 0:
            n_chains = 20
            n_samples = 10
            rng = np.random.RandomState(123)
            test_set_x = test_set[0]
            number_of_test_samples = test_set_x.shape[0]
            test_set_x = theano.shared( value = np.asarray(test_set_x, dtype=theano.config.floatX),
                                        name = 'test', borrow = True)

            # pick random test examples, with which to initialize the persistent chain
            test_idx = rng.randint(number_of_test_samples - n_chains)
            persistent_vis_chain = theano.shared(
                np.asarray(
                    test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
                    dtype=theano.config.floatX
                )
            )
            # end-snippet-6 start-snippet-7
            plot_every = 10000
            # define one step of Gibbs sampling (mf = mean-field) define a
            # function that does `plot_every` steps before returning the
            # sample for plotting
            (
                [
                    presig_hids,
                    hid_mfs,
                    hid_samples,
                    presig_vis,
                    vis_mfs,
                    vis_samples
                ],
                updates
            ) = theano.scan(
                mpf_optimizer.gibbs_vhv,
                outputs_info=[None, None, None, None, None, persistent_vis_chain],
                n_steps=plot_every
            )

            # add to updates the shared variable that takes care of our persistent
            # chain :.
            updates.update({persistent_vis_chain: vis_samples[-1]})
            # construct the function that implements our persistent chain.
            # we generate the "mean field" activations for plotting and the actual
            # samples for reinitializing the state of our persistent chain
            sample_fn = theano.function(
                [],
                [
                    vis_mfs[-1],
                    vis_samples[-1]
                ],
                updates=updates,
                name='sample_fn'
            )

            # create a space to store the image for plotting ( we need to leave
            # room for the tile_spacing as well)
            image_data = np.zeros(
                (29 * n_samples + 1, 29 * n_chains - 1),
                dtype='uint8'
            )
            for idx in range(n_samples):
                # generate `plot_every` intermediate samples that we discard,
                # because successive samples in the chain are too correlated
                vis_mf, vis_sample = sample_fn()
                print(' ... plotting sample ', idx)
                image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
                    X=vis_mf,
                    img_shape=(28, 28),
                    tile_shape=(1, n_chains),
                    tile_spacing=(1, 1)
                )

            # construct image
            image = Image.fromarray(image_data)
            image.save(path + '/samples_%i.eps' % em_epoch)
            # end-snippet-7
            # os.chdir('../')


    loss_savename = path + '/train_loss.eps'
    show_loss(savename= loss_savename, epoch_error= mean_epoch_error)

    end_time = timeit.default_timer()

    running_time = (end_time - start_time)

    print ('Training took %f minutes' % (running_time / 60.))

    return saveName_w, saveName_b


if __name__ == '__main__':


    learning_rate_list = [0.001]
    # hyper-parameters are: learning rate, num_samples, sparsity, beta, epsilon, batch_sz, epoches
    # Important ones: num_samples, learning_rate,
    hidden_units_list = [196]
    n_samples_list = [1]
    beta_list = [0]
    sparsity_list = [.1]
    batch_list = [40]
    decay_list = [0.0001,0, 0.001]

    for batch_size in batch_list:
        for n_samples in n_samples_list:
            for hidden_units in hidden_units_list:
                for decay in decay_list:
                    for learning_rate in learning_rate_list:
                            savename_w, savename_b = em_mpf(hidden_units = hidden_units,learning_rate = learning_rate, epsilon = 0.01,decay=decay,
                                   batch_sz=batch_size, explicit_EM=False)












