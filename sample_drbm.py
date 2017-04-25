import numpy as np
from utils_mpf import tile_raster_images
import gzip, pickle, os
import theano
import Image
from utils_mpf import sigmoid


dataset = 'mnist.pkl.gz'
f = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
f.close()

path = '../mpf_results/lay1_196'
if not os.path.exists(path):
    os.makedirs(path)

def generate_from_rbm(W_file, b_file, W2_file, b2_file):



    W = np.load(W_file)
    print(W.shape)
    b = np.load(b_file)
    visible_units = W.shape[0]

    b_vis = b[:visible_units].reshape([1,-1])
    b_hid = b[visible_units:].reshape([1,-1])


    W2 = np.load(W2_file)
    print(W.shape)
    b2 = np.load(b2_file)
    visible2_units = W2.shape[0]

    b2_vis = b2[:visible2_units].reshape([1,-1])
    b2_hid = b2[visible2_units:].reshape([1,-1])

    n_chains = 20
    n_samples = 10
    rng = np.random.RandomState(123)
    test_set_x = test_set[0]
    number_of_test_samples = test_set_x.shape[0]

    # pick random test examples, with which to initialize the persistent chain
    # test_idx = rng.randint(number_of_test_samples - n_chains)
    # persistent_vis_chain = np.asarray(test_set_x[test_idx:test_idx + n_chains])
    # print(test_set[1][test_idx:test_idx + n_chains])



    # end-snippet-6 start-snippet-7
    plot_every = 8
    image_data = np.zeros(
        (29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8'
    )

    for idx in range(n_samples):
        persistent_vis_chain = np.random.randint(2,size=(n_chains,100))
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        v_samples = persistent_vis_chain
        down_sample2 = None
        # for j in range(plot_every):
        #     ### Experiment shows that
        #
        #     upact = sigmoid(np.dot(v_samples,W) + b_hid)
        #     up_sample = np.random.binomial(n=1, p= upact)
        #
        #     upact2 = sigmoid(np.dot(up_sample,W2)+b2_hid)
        #     up2_sample = np.random.binomial(n=1,p=upact2)
        #
        #     downact1 = sigmoid(np.dot(up2_sample, W2.T) + b2_vis)
        #     down_sample1 = np.random.binomial(n=1, p= downact1)
        #
        #     downact2 = sigmoid(np.dot(down_sample1,W.T)+b_vis)
        #     v_samples = np.random.binomial(n=1,p=downact2)



        for j in range(plot_every):
            ### Experiment shows that

            # upact = sigmoid(np.dot(v_samples,W) + b_hid)
            # up_sample = np.random.binomial(n=1, p= upact)
            #
            # upact2 = sigmoid(np.dot(up_sample,W2)+b2_hid)
            # up2_sample = np.random.binomial(n=1,p=upact2)

            downact1 = sigmoid(np.dot(v_samples, W2.T) + b2_vis)
            down_sample1 = np.random.binomial(n=1, p= downact1)

            upact1 = sigmoid(np.dot(down_sample1,W2)+b2_hid)
            v_samples = np.random.binomial(n=1,p=upact1)


        print(down_sample1.shape)

            # vis_mf = sigmoid(np.dot(down_sample1, W.T) + b_vis)
            # v_samples = np.random.binomial(n=1,p=vis_mf)

        for j in range(plot_every):

            #v_samples = vis_mf

            downact2 = sigmoid(np.dot(down_sample1, W.T) + b_vis)
            down_sample2 = np.random.binomial(n=1,p=downact2)

            upact2 = sigmoid(np.dot(down_sample2, W) + b_hid)
            down_sample1 = np.random.binomial(n=1,p=upact2)


        print(' ... plotting sample ', idx)
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
            X= downact2,
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

        # image_binary_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
        #     X=v_samples,
        #     img_shape=(28, 28),
        #     tile_shape=(1, n_chains),
        #     tile_spacing=(1, 1)
        # )

    # construct image
    image = Image.fromarray(image_data)
    # image_binary = Image.fromarray(image_binary_data)
    image.save(path + '/samples.png')
    # image_binary.save(path + '/binary_samples.eps')



decay_list = [0.0001]
lr_list = [0.001]
weight1 = '../mpf_results/lay1_196/weights_499.npy'
bias1 = '../mpf_results/lay1_196/bias_499.npy'
perplexity = 20

for decy in decay_list:
    for lr in lr_list:
        weight2 = '../mpf_results/lay1_196/lay2_100/decay_' + str(decy) + '/lr_' + str(lr) + '/weights_399.npy'
        bias2 = '../mpf_results/lay1_196/lay2_100/decay_' + str(decy) + '/lr_' + str(lr) + '/bias_399.npy'



        generate_from_rbm(W_file=weight1, b_file=bias1,W2_file= weight2, b2_file=bias2)
