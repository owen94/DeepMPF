import numpy as np
from utils_mpf import *
from PIL import Image


visible = 784
hidden_1 = 196

#path_w = '../DBM_results/Samples/196_64/weight_599.npy'
#path_b = '../DBM_results/Samples/196_64/bias_599.npy'

path_w = '../mpf_results/196/weights_499.npy'
path_b = '../mpf_results/196/bias_499.npy'


base_path_w = '../rbm_base/weights_395.npy'
base_path_b = '../rbm_base/bias_395.npy'


dataset = 'mnist.pkl.gz'
f = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
f.close()

binarizer = preprocessing.Binarizer(threshold=0.5)
training_data =  binarizer.transform(train_set[0])
test_data = test_set[0]

def sampling_rbm(W_file, b_file, train_data, n_steps = 5):
    # We sample from a single RBM and compute the log-likelihood
    n_sample = 10000

    # W = np.load(W_file)
    # b = np.load(b_file)
    #
    # #W = W[:visible,visible:]
    # b0 = b[:visible]
    # b1 = b[visible:]


    # This is for the rbm baseline
    W = np.load(W_file)
    b = np.load(b_file)
    b0 = b[0]
    b1 = b[1]

    activation = sigmoid(np.dot(train_data,W) + b1)
    mean_h = np.mean(activation, axis=0)
    std_h = np.std(activation,axis=0)

    #initial = np.random.normal(loc=mean_h,scale=std_h,size=(10000, 196))
    downact = None
    initial = np.random.binomial(n=1, p= mean_h, size=(n_sample, 196))
    for i in range(n_steps):
        downact = sigmoid(np.dot(initial,W.T) + b0)
        down_sample = np.random.binomial(n=1, p= downact)
        upact = sigmoid(np.dot(down_sample,W)+b1)
        up_samples = np.random.binomial(n=1,p=upact)
        initial = up_samples

    return downact


def for_gpu_sample(W, b0, b1, train_data, n_steps = 5):
    # We sample from a single RBM and compute the log-likelihood
    n_sample = 10000

    # This is for the rbm baseline
    activation = sigmoid(np.dot(train_data,W) + b1)
    mean_h = np.mean(activation, axis=0)
    #initial = np.random.normal(loc=mean_h,scale=std_h,size=(10000, 196))
    downact = None
    initial = np.random.binomial(n=1, p= mean_h, size=(n_sample, 196))
    for i in range(n_steps):
        downact = sigmoid(np.dot(initial,W.T) + b0)
        down_sample = np.random.binomial(n=1, p= downact)
        upact = sigmoid(np.dot(down_sample,W)+b1)
        up_samples = np.random.binomial(n=1,p=upact)
        initial = up_samples

    return downact


def sampling_dbm(W_file, b_file, n_rbm = 2):

    dataset = 'mnist.pkl.gz'
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
    f.close()

    binarizer = preprocessing.Binarizer(threshold=0.5)
    training_data =  binarizer.transform(train_set[0])
    test_data = test_set[0]

    n_sample = 10000


    W = np.load(W_file)
    b = np.load(b_file)

    if n_rbm == 2:
        hidden_list = [784, 196, 64]
    else:
        hidden_list = [784, 196, 196, 64]




def parzen(x, mu, sigma):
    '''
    :param x: a single sample need to be tested for likelihood computing
    :param mu: the generated sample used to compute the model distribution
    :param sigma: the parzen windom bandwidth
    :return: likelihood value
    '''
    in_kernel = (x - mu)/sigma
    b = -0.5 * np.sum((in_kernel**2),axis= 1)
    exp_b = np.exp(b - np.max(b))
    log_lld_1 = np.log(np.mean(exp_b)) + np.max(b)
    z = mu.shape[1] * np.log(np.sqrt(np.pi * 2)*sigma)

    log_lld = log_lld_1 - z

    return log_lld

def batch_parzen(x, mu, sigma, batch_sz = 20):
    '''
    :param x: the whole test data set instead of a single sample
    :param mu: generated samples
    :param sigma: bandwidth
    :return: the mean log-likelihood
    '''
    num_test = x.shape[0]
    num_batches = num_test // batch_sz
    lld = []

    for i in range(num_batches):
        batch = x[i * batch_sz : (i+1)*batch_sz,:]
        batch = np.transpose(a=batch[np.newaxis,:],axes=(1,0,2))
        mu_1 = mu[np.newaxis,:]
        a = (batch - mu_1)/sigma
        b = -0.5 * (a**2).sum(2)
        max_b = np.max(b,axis=1)
        log_lld_1 = np.log(np.mean(np.exp(b - np.transpose(max_b[np.newaxis,:])),axis=1)) + max_b
        z = mu.shape[1] * np.log(np.sqrt(np.pi * 2)*sigma)
        log_lld = log_lld_1 - z
        lld.extend(log_lld)

    return np.mean(np.array(lld))


def gpu_parzen(mu, sigma):
    x = T.matrix()
    mu = theano.shared( value = np.asarray(mu, dtype=theano.config.floatX))
    a = ( x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1) ) / sigma
    b = -0.5*(a**2).sum(2)
    max_ = b.max(1)
    log_lld_1 = max_ + T.log(T.exp(b - max_.dimshuffle(0, 'x')).mean(1))
    Z = mu.shape[1] * T.log(sigma * T.sqrt(np.pi * 2))
    return theano.function([x], log_lld_1 - Z)

def get_ll(x, gpu_parzen, batch_size=10): # get parzen window log-likelihood

    inds = range(x.shape[0])
    n_batches = int(np.ceil(float(len(inds)) / batch_size))
    lls = []
    for i in range(n_batches):
        ll = gpu_parzen(x[inds[i::n_batches]])
        lls.extend(ll)

    return lls


def compute_lls():
    mu = sampling_rbm(W_file= base_path_w,b_file= base_path_b)
    print(mu.shape)
    print(test_data.shape)

    image_data = np.zeros(
        (29 * 1 + 1, 29 * 10 - 1), dtype='uint8'
    )
    image_data[29 * 0:29 * 0 + 28, :] = tile_raster_images(
            X= mu[:10,:],
            img_shape=(28, 28),
            tile_shape=(1, 10),
            tile_spacing=(1, 1)
        )

    image = Image.fromarray(image_data)
    image.show()

    sigma = [0.2]
    num_test = test_data.shape[0]
    log_lld = 0

    for s in sigma:
        log_lld = 0
        for i in range(num_test):
            lld = parzen(x=test_data[i,:], mu= mu, sigma=s)
            #print('the lld for the {}th sample is {}'.format(i, lld))
            log_lld += lld
        print('the lld for sigma {}  is {}'.format(s, log_lld/num_test))
        lld = batch_parzen(x=test_data,mu=mu,sigma=s)
        print(lld)

        # lld = get_ll(test_data, gpu_parzen(mu, sigma), batch_size=10)
        # print( np.mean(np.array(lld))  )
















