
# Towards Biologically Plausible Deep Learning

import theano, pickle, time, os
import theano.tensor as T
import numpy as np
from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams

def castX(x) : return theano._asarray(x, dtype=theano.config.floatX)
def sharedX(x) : return theano.shared( theano._asarray(x, dtype=theano.config.floatX) ) 
def randn(shape,mean,std) : return sharedX( mean + std * np.random.standard_normal(size=shape) )
def rand(shape, irange) : return sharedX( - irange + 2 * irange * np.random.rand(*shape) )
def zeros(shape) : return sharedX( np.zeros(shape) ) 

def rand_ortho(shape, irange) : # random orthogonal matrixp
    A = - irange + 2 * irange * np.random.rand(*shape)
    U, s, V = np.linalg.svd(A, full_matrices=True)
    return sharedX(  np.dot(U, np.dot( np.eye(U.shape[1], V.shape[0]), V )) )

def one_hot(labels, nC=None): # make one-hot code
    nC = np.max(labels) + 1 if nC is None else nC
    code = np.zeros( (len(labels), nC), dtype='float32' )
    for i,j in enumerate(labels) : code[i,j] = 1.
    return code

def sigm(x) : return T.nnet.sigmoid(x)
def sfmx(x) : return T.nnet.softmax(x)
def tanh(x) : return T.tanh(x)
def sign(x) : return T.switch(x > 0., 1., -1.)

def relu(x) : return T.switch(x > 0., x, 0.)
def softplus(x) : return T.nnet.softplus(x)

def mse(x,y) : return T.sqr(x-y).sum(axis=1).mean() # mean squared error

RNG = MRG_RandomStreams(max(np.random.RandomState(1364).randint(2 ** 15), 1))

def samp(x) : # x in [0,1] sampling binary values from probs
    rand = RNG.uniform(x.shape, ndim=None, dtype=None, nstreams=None)
    return T.cast( rand < x, dtype='floatX');

# gaussian corruption
def gaussian(x, std, rng=RNG) : return x + rng.normal(std=std, size=x.shape, dtype=x.dtype)

def rms_prop( param_grad_dict, learning_rate, 
                    momentum=.9, averaging_coeff=.95, stabilizer=.0001) :
    updates = OrderedDict()
    for param in param_grad_dict.keys() :

        inc = sharedX(param.get_value() * 0.)
        avg_grad = sharedX(np.zeros_like(param.get_value()))
        avg_grad_sqr = sharedX(np.zeros_like(param.get_value()))

        new_avg_grad = averaging_coeff * avg_grad \
            + (1 - averaging_coeff) * param_grad_dict[param]
        new_avg_grad_sqr = averaging_coeff * avg_grad_sqr \
            + (1 - averaging_coeff) * param_grad_dict[param]**2

        normalized_grad = param_grad_dict[param] / \
                T.sqrt(new_avg_grad_sqr - new_avg_grad**2 + stabilizer)
        updated_inc = momentum * inc - learning_rate * normalized_grad

        updates[avg_grad] = new_avg_grad
        updates[avg_grad_sqr] = new_avg_grad_sqr
        updates[inc] = updated_inc
        updates[param] = param + updated_inc

    return updates

def get_ll(x, parzen, batch_size=10): # get parzen window log-likelihood

    inds = range(x.shape[0])
    n_batches = int(np.ceil(float(len(inds)) / batch_size))

    times = []
    lls = []
    for i in range(n_batches):
        begin = time.time()
        ll = parzen(x[inds[i::n_batches]])
        end = time.time()
        times.append(end-begin)
        lls.extend(ll)

        #if i % 10 == 0:
        #    print i, numpy.mean(times), numpy.mean(nlls)

    return np.array(lls)


def log_mean_exp(a):
    max_ = a.max(1)
    return max_ + T.log(T.exp(a - max_.dimshuffle(0, 'x')).mean(1))


def theano_parzen(mu, sigma): # make parzen window

    x = T.matrix()
    mu = theano.shared(mu)
    a = ( x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1) ) / sigma
    E = log_mean_exp(-0.5*(a**2).sum(2))
    Z = mu.shape[1] * T.log(sigma * np.sqrt(np.pi * 2))

    return theano.function([x], E - Z)



# load MNIST data into shared variables 
# train data 50000x784, label 50000x1
# valid data 10000x784, label 10000x1
# test  data 10000x784, label 10000x1
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = \
        np.load('/path/to/mnist.pkl')
np_test_x, np_valid_x = test_x, valid_x

train_x, train_y, valid_x, valid_y, test_x, test_y = \
    sharedX(train_x), sharedX(one_hot(train_y)), \
    sharedX(valid_x), sharedX(one_hot(valid_y)), \
    sharedX(test_x),  sharedX(one_hot(test_y ))

def exp(__lr) :

    max_epochs, batch_size, n_batches = 1000, 100, 500 # = 50000/100
    nX, nH1, nH2 = 784, 1000, 100

    W1 = rand_ortho((nX,  nH1), np.sqrt(6./(nX +nH1)));  B1 = zeros((nH1,))
    W2 = rand_ortho((nH1, nH2), np.sqrt(6./(nH1+nH2)));  B2 = zeros((nH2,))

    V1 = rand_ortho((nH1,  nX), np.sqrt(6./(nH1+ nX)));  C1 = zeros((nX, ))
    V2 = rand_ortho((nH2, nH1), np.sqrt(6./(nH2+nH1)));  C2 = zeros((nH1,))

    # layer definitions - functions of layers
    F1 = lambda x  : softplus( T.dot( x,  W1 ) + B1 )
    G1 = lambda h1 : sigm( T.dot( h1, V1 ) + C1 )

    F2 = lambda h1 : sigm( T.dot( h1, W2 ) + B2 )
    G2 = lambda h2 : softplus( T.dot( h2, V2 ) + C2 )

    i, e = T.lscalar(), T.fscalar(); X, Y = T.fmatrices(2)

    givens_train = lambda i : { X : train_x[ i*batch_size : (i+1)*batch_size ],  
                                Y : train_y[ i*batch_size : (i+1)*batch_size ] }
    givens_valid, givens_test = { X : valid_x, Y : valid_y }, { X : test_x, Y : test_y  }
    givens_empty = { X : sharedX(np.zeros((10000,784))), Y : sharedX(np.zeros((10000,10))) }

    def iteration(X, k, alpha, beta = 0.01) : # infer h1 and h2 from x
        H1 = F1(X); H2 = F2(H1)
        for i in range(k) : 
            H2 = H2 + alpha*( F2(H1) - F2(G2(H2)) )
            H1 = H1 + alpha*( F1(X)  - F1(G1(H1)) ) + alpha*beta*( G2(H2) - H1 )
        return H1, H2

    H1, H2 = F1(X), F2(F1(X))
    H1_, H2_ = iteration(X, 15, 0.1)

    def avg_bin(x, k) : # average of sampled random binary values
        S = 0.*x
        for i in range(k) : S = S + samp(x)
        return S / k

    # get gradients
    g_V1, g_C1 = T.grad( mse( G1(gaussian(H1_,0.3)), X   ), [V1, C1], consider_constant=[H1_, X] )
    g_W1, g_B1 = T.grad( mse( F1(gaussian(X  ,0.5)), H1_ ), [W1, B1], consider_constant=[X, H1_] )

    g_V2, g_C2 = T.grad( mse( G2( avg_bin(H2_,3) ),  H1_ ), [V2, C2], consider_constant=[H2_, H1_] )
    g_W2, g_B2 = T.grad( mse( F2(gaussian(H1_,0.5)), H2_ ), [W2, B2], consider_constant=[H1_, H2_] )

    cost = mse(  G1(G2(F2(F1(X)))),  X  )

    # training
    train_sync = theano.function( [i,e], [cost], givens = givens_train(i), on_unused_input='ignore', 
        updates=rms_prop( { W1 : g_W1, B1 : g_B1, V1 : g_V1, C1 : g_C1,
                            W2 : g_W2, B2 : g_B2, V2 : g_V2, C2 : g_C2 }, __lr ) )

    def get_samples() : # get samples from the model
        X, Y = T.fmatrices(2)
        givens_train_samples = { X : train_x[0:50000], Y : train_y[0:50000] }

        H1, H2 = iteration(X, 15, 0.1)

        # get prior statistics (100 mean and std) 
        H2_mean = T.mean( H2, axis=0 ); H2_std = T.std( H2, axis=0 )
        # sampling h2 from prior
        H2_ = RNG.normal( (10000,100), avg=H2_mean, std=4*H2_std, ndim=None, dtype=H2.dtype, nstreams=None)

        # iterative sampling from samples h2
        X_ = G1(G2(H2_))
        for i in range(3) : 
            H1_, H2_ = iteration(X_, 15, 0.1, 3)
            X_ = G1(H1_)
        #H1_, H2_ = iteration(X_, 1, 0.1, 3)
        #X_ = G1(H1_)

        sampling = theano.function([], X_, on_unused_input='ignore', givens = givens_train_samples)
        samples = sampling() 
        np.save('samples',samples)
        return samples

    # get test log-likelihood
    def test_ll(sigma) :
        samples = get_samples()
        return get_ll(np_test_x, theano_parzen(samples, sigma), batch_size=10)

    test_cost = theano.function([i,e], [cost], on_unused_input='ignore', givens=givens_test  )

    print('epochs test_loglikelihood time')

    # training loop
    t = time.time(); monitor = { 'train' : [], 'valid' : [], 'test' : [], 'test_ll':[], 'test_ll_base':[] }
    for e in range(1,max_epochs+1) :
        monitor['train'].append(  np.array([ train_sync(i,e) for i in range(n_batches) ]).mean(axis=0)  )

        if e % 5 == 0 :
            monitor['test'].append( test_cost(0,0) )
            monitor['test_ll'].append( np.mean(test_ll(0.2)) )
            print(e, monitor['test_ll'][-1], time.time() - t)


exp(0.00001)










