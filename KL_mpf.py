import gzip
import timeit, pickle, sys, math, os
import sys
sys.setrecursionlimit(40000)
from scipy.optimize import fmin_l_bfgs_b as minimize
from sklearn import preprocessing
from utils_mpf import *



def unravel_params(theta,visible_size):
    W = theta[0:visible_size*visible_size].reshape(visible_size,visible_size)
    b = theta[visible_size*visible_size:].reshape(visible_size,1)
    return W,b

def ravel_params(W,b):
    return np.concatenate((W.ravel(),b.ravel()))


def sigmoid(x):
    return 1/(1+np.exp(-x))


def normalizeData(patches):
    # Squash data to [0.1, 0.9] since we use sigmoid as the activation
    # function in the output layer

    # Remove DC (mean of images).
    patches = patches-np.array([np.mean(patches,axis=1)]).T

    # Truncate to +/-3 standard deviations and scale to -1 to 1
    pstd = 3*np.std(patches)
    patches = np.fmax(np.fmin(patches,pstd),-pstd)/pstd

    # Rescale from [-1,1] to [0.1,0.9]
    patches = (patches+1)*0.4+0.1
    return patches

class KL_dmpf_optimizer(object):

    def __init__(self, vis_units, hid_units,epsilon, batch_sz = 20):

        self.visible_units = vis_units
        self.hidden_units = hid_units
        self.epsilon = epsilon
        self.batch_sz = batch_sz
        self.num_neurons = self.visible_units + self.hidden_units
        numpy_rng = np.random.RandomState(12336)
        self.W = get_mpf_params(self.visible_units,self.hidden_units)
        self.b = np.zeros(self.num_neurons)
        self.zero_grad = None

    def propup(self, data_samples):

        activations =  sigmoid(np.dot(data_samples,self.W[:self.visible_units,self.visible_units:])
                               +self.b[self.visible_units:].reshape([1,-1]))
        samples = np.random.binomial(n=1,p= activations)
        return [activations, samples]

    def get_samples(self, theta, data_samples, n_samples = 5):

        self.W, self.b = unravel_params(theta, visible_size= self.num_neurons)


        if self.zero_grad is None:
            a = np.ones((self.visible_units,self.hidden_units))
            b = np.zeros((self.visible_units,self.visible_units))
            c = np.zeros((self.hidden_units,self.hidden_units))
            zero_grad_u = np.concatenate((b,a),axis = 1)
            zero_grad_d = np.concatenate((a.T,c),axis=1)
            self.zero_grad = np.concatenate((zero_grad_u,zero_grad_d),axis=0)


        sample_prob = None
        data = None
        norm_sample_prob = None

        #rho = []

        for i in range(n_samples):
            prop = self.propup(data_samples=data_samples)
            activations = prop[0]
            hidden_samples = prop[1]
            new_data = np.concatenate((data_samples,hidden_samples),axis=1)

            probability = (1 - hidden_samples) * (1-activations) + hidden_samples * activations
            new_sample_prob = np.prod(probability, axis = 1)

            if data is None:
                data = new_data
                sample_prob = new_sample_prob
                norm_sample_prob = new_sample_prob
               # rho = activations
            else:
                data = np.concatenate((data,new_data),axis = 0)
                sample_prob = np.concatenate((sample_prob,new_sample_prob))
                norm_sample_prob += new_sample_prob
                #rho = np.concatenate((rho,activations),axis = 0)

        norm_sample_prob = np.tile(norm_sample_prob,reps=n_samples)
        sample_prob = sample_prob / norm_sample_prob

        return sample_prob, data, #rho


    def get_cost_updates(self, theta, data, sample_prob, epsilon, sparsityParam =0.1, beta =0 ):


         # computing the cost and gradient
        self.W, self.b = unravel_params(theta, visible_size= self.num_neurons)

        if epsilon != self.epsilon:
            self.epsilon = epsilon

        self.batch_sz = data.shape[0]


        z = 1/2 - data
        energy_difference = z * (np.dot(data,self.W)+ self.b.reshape([1,-1]))

        sample_prob = sample_prob.reshape((1,-1)).T
        k = np.asarray(np.ones(self.num_neurons)).reshape([1,-1])
        sample_prob = np.dot(sample_prob, k)

        cost = (self.epsilon/self.batch_sz) * np.sum(np.exp(energy_difference)* sample_prob)

        # compute the KL-divergence and gradient

        raw_samples = data[:,:self.visible_units]
        activations = self.propup(raw_samples)[0]
        rho = np.mean(activations,axis=0)

        KL = beta*((-sparsityParam/rho) + ((1 - sparsityParam)/(1 - rho)))

        KL_b_grad = KL*activations*(1-activations)

        KL_grad = np.dot(raw_samples.T, KL_b_grad)

        cost_kl = beta*np.sum(sparsityParam*np.log(sparsityParam/rho)+
                              (1 - sparsityParam)*np.log((1 - sparsityParam)/(1 - rho)))

        a = np.zeros((self.visible_units,self.visible_units))
        b = np.zeros((self.hidden_units,self.hidden_units))
        kl_grad1 = np.concatenate((a, KL_grad),axis=1)
        kl_grad2 = np.concatenate((KL_grad.T, b),axis=1)
        kl_grad = np.concatenate((kl_grad1,kl_grad2),axis=0)


       ###  add  decay weight  #############
        # decay = 0.0001
        # cost_weight = 0.5 * decay * np.sum(self.W**2)
        #
        # grad_decay = decay*self.W
        # computing the gradient

        cost += cost_kl

        #cost += cost_weight

        h = z * np.exp(energy_difference) * sample_prob

        bgrad = np.mean(h,axis=0)*self.epsilon
        Wgrad = (np.dot(h.T,data)+np.dot(data.T,h))*self.epsilon/self.batch_sz


        #np.fill_diagonal(Wgrad,0)
        Wgrad *= self.zero_grad

        Wgrad += (kl_grad/self.batch_sz)
        #Wgrad += grad_decay
        bgrad[self.visible_units:] += np.mean(KL_b_grad,axis=0)

        grad = ravel_params(Wgrad,bgrad)

        return cost, grad


def computeNumericalGradient(J,theta):
    # numgrad = computeNumericalGradient(J, theta)
    # theta: a vector of parameters
    # J: a function that outputs r.
    # Calling y = J(theta) will return the function value at theta.

    # Initialize numgrad with zeros
    numgrad = np.zeros(np.shape(theta))

    ## ---------- YOUR CODE HERE --------------------------------------
    # Instructions:
    # Implement numerical gradient checking, and return the result in numgrad.
    # (See Section 2.3 of the lecture notes.)
    # You should write code so that numgrad(i) is (the numerical approximation to) the
    # partial derivative of J with respect to the i-th input argument, evaluated at theta.
    # I.e., numgrad(i) should be the (approximately) the partial derivative of J with
    # respect to theta(i).
    #
    # Hint: You will probably want to compute the elements of numgrad one at a time.
    for i in range(0,numgrad.shape[0]):
        k = np.zeros(np.shape(theta))
        k[i] = 0.0001
        epsW,epsb = unravel_params(k,20)
        epsW = epsW+epsW.T
        k = ravel_params(epsW,epsb)
        y1 = J(theta + k)
        y2 = J(theta - k)
        numgrad[i] = (y1-y2)/0.0002

    return numgrad


def load_mnist(dataset = None):

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

    return data


def load_IMAGE():

    Images = np.load('IMAGES.npy')
    num_patches = 10000
    patch_size = 8
    patches = np.zeros((num_patches,patch_size*patch_size))

    for i in range(num_patches):
        pix_index = np.random.randint(500,size=2)
        im_index = np.random.randint(10)
        patches[i,:] = patches[i,:] + \
                       Images[pix_index[0]:pix_index[0]+patch_size,pix_index[1]:pix_index[1]+patch_size,
                       im_index].reshape(1,patch_size*patch_size)


    patches = normalizeData(patches)
    #display(patches[:100,:])

    binarizer = preprocessing.Binarizer(threshold=0.5)
    patches =  binarizer.transform(patches)

    return patches

def train_bfgs_rbm(epsilon,n_samples,epoches):

    vis_units = 64
    hid_units = 16
    beta = 3
    sparsity = 0.1
    epsilon = epsilon
    n_samples = n_samples

    path = '../Grid_Patch_filters/num_samples_' + str(n_samples)
    if not os.path.exists(path):
        os.makedirs(path)



    #  # ###### Check Gradient #############################################
    # rbm_data = np.load('rbm_samples_10000.npy')
    # print(rbm_data.shape)
    # mpf_optimizer = KL_dmpf_optimizer(vis_units = vis_units, hid_units= hid_units, epsilon = epsilon)
    # theta = ravel_params(mpf_optimizer.W, mpf_optimizer.b)
    # sample_prob, data = mpf_optimizer.get_samples(theta=theta, data_samples = rbm_data, n_samples = n_samples)
    #
    #
    # # numpy_rng = np.random.RandomState(555555)
    # # W = numpy_rng.randn(20,20)/np.sqrt(20*20)
    # # W = (W + W.T)/2
    # # np.fill_diagonal(W,0)
    # # b = np.zeros((20,1))
    # # theta = ravel_params(W, b)
    # numgrad = computeNumericalGradient(lambda x: mpf_optimizer.get_cost_updates(
    #     x, data, sample_prob,epsilon)[0],theta)
    # cost, grad = mpf_optimizer.get_cost_updates(theta,data, sample_prob,epsilon)
    # # Use this to visually compare the gradients side by side
    # print(np.array([numgrad,grad]).T)
    # # Compare numerically computed gradients with the ones obtained from backpropagation
    # diff = norm(numgrad-grad)/norm(numgrad+grad)
    # print(diff)
    # print('diff is computed')
    ###########################################################





    patches = load_IMAGE()
    #patches = load_mnist()

    #print(patches[0,:])

    mpf_optimizer = KL_dmpf_optimizer(vis_units = vis_units, hid_units= hid_units, epsilon = epsilon)

    theta = ravel_params(mpf_optimizer.W, mpf_optimizer.b)

    sample_prob, data = mpf_optimizer.get_samples(theta=theta, data_samples = patches, n_samples = n_samples)


    #####  Run the MPF ######

    # We first try the MPF with l-bfgs algorithms

    opttheta,cost,messages = minimize(mpf_optimizer.get_cost_updates,theta,fprime=None,maxiter=400,
                                     args=(data, sample_prob,epsilon))

    for i in range(epoches):
        theta = opttheta
        sample_prob, data = mpf_optimizer.get_samples(theta= theta, data_samples = patches, n_samples=n_samples)
        opttheta,cost,messages = minimize(mpf_optimizer.get_cost_updates,theta,fprime=None,maxiter=400,
                                         args=(data, sample_prob,epsilon,sparsity,beta))
        print('The cost for dmpf in epoch %d is %f'% (i, cost))

        W1,b1 = unravel_params(opttheta,visible_size= vis_units+hid_units)
        W1 = W1[:vis_units,vis_units:]
        saveName = path + '/weights_' + str(epsilon) + '_' + str(i) + '.png'
        display(W1.T,saveName=saveName)

    return opttheta



if __name__ == '__main__':


    # epsilon_list = [#0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    #                 0.01,0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
    #                 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
    #                 0.0001, 0.0002, 0.0003, 0.0004,0.0005, 0.0006, 0.0007, 0.008, 0.0009]
    #
    #
    #
    # for epsilon in epsilon_list:
    #     for n_samples in range(10):
    #         train_bfgs_rbm(epsilon= epsilon,  n_samples= n_samples+2 , epoches=20 )

    theta = train_bfgs_rbm(epsilon= 0.01,  n_samples= 5, epoches=7)











