'''
Give a trained model, this will test the imputation performance..
'''

import numpy as np
import gzip, pickle, random
from sklearn import preprocessing
import matplotlib.pyplot as plt
from utils_mpf import  *
from PIL import Image


class get_samples(object):

    def __init__(self, hidden_list = [], W = None, b = None):
        self.hidden_list = hidden_list
        self.num_rbm = len(hidden_list) - 1
        self.W = W
        self.b = b

    def propup(self, i, data):
        if len(self.W) >= 1:
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


def element_corruption(data, corrupt_row = 2):
    row = random.sample(range(10,18), 1)[0]
    x =  np.random.binomial(n=1, p= 0.5, size=(data.shape[0], corrupt_row * 28))
    data[:,row*28:(row+corrupt_row)*28]  += x
    binarizer = preprocessing.Binarizer(threshold=0.99)
    data =  binarizer.transform(data)
    return np.array(data)

def top_corruption(data, corrupt_row = 10, start = 4):
    x = np.random.binomial(n=1,p=0.5, size=(data.shape[0], 28*corrupt_row))
    data[:,28*start : start *28 + 28*corrupt_row] += x
    binarizer = preprocessing.Binarizer(threshold=0.99)
    data =  binarizer.transform(data)
    return np.array(data)

def bottom_corruption(data, corrupt_row = 10, start = 0):
    x = np.random.binomial(n=1,p=0.5, size=(data.shape[0], 28*corrupt_row))
    data[:,28*(28-corrupt_row - start): 28*(28-start)] += x
    binarizer = preprocessing.Binarizer(threshold=0.99)
    data =  binarizer.transform(data)
    return np.array(data)

def left_corruption(data, corrupt_row = 10):
    for i in range(28):
        x = np.random.binomial(n=1, p=0.5, size=(data.shape[0], corrupt_row))
        data[:,28*i : 28*i + corrupt_row] += x
    binarizer = preprocessing.Binarizer(threshold=0.99)
    data =  binarizer.transform(data)
    return np.array(data)

def right_corruption(data, corrupt_row = 10):
    for i in range(28):
        x = np.random.binomial(n=1, p=0.5, size=(data.shape[0], corrupt_row))
        data[:,28*(i+1) - corrupt_row : 28*(i+1)] += x
    binarizer = preprocessing.Binarizer(threshold=0.99)
    data =  binarizer.transform(data)
    return np.array(data)


def reconstruct(activations, W, b, corrupt_row):

    for idx in range(n_samples):
        persistent_vis_chain = np.random.binomial(n=1, p= activations, size=activations.shape)

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
    return downact1
    # image = Image.fromarray(image_data)
    # image.save(savepath1 + '/recover.eps')



############################################################
path_w = '../LLD/DBM_196_196_64/decay_1e-05/lr_0.001/weight_199.npy'
path_b = '../LLD/DBM_196_196_64/decay_1e-05/lr_0.001/bias_199.npy'
savepath1 = '../LLD/Samples/'


W = np.load(path_w)
b = np.load(path_b)
W = [W[0]]
b = [ b[0] ]
hidden_list = [784, 196]


# hidden_list = [784, 196, 196, 64]

num_rbm = len(hidden_list) -1
n_chains = 8
n_samples = 1
plot_every = 20
image_data = np.zeros(
    (29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8'
)

dataset = 'mnist.pkl.gz'
f = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
f.close()
#print(train_set[0][0])
binarizer = preprocessing.Binarizer(threshold=0.5)
data =  binarizer.transform(train_set[0])

img_index = random.sample(range(0, data.shape[0]), n_chains)
print(img_index)
img_data = data[img_index,:]

print(train_set[1][img_index])


##################################################

corrupt_row = 12
#cor = top_corruption(img_data, corrupt_row=8)
#cor = top_corruption(img_data, corrupt_row= 8)
# corruption_type = 'top'
# cor = top_corruption(img_data, corrupt_row= corrupt_row, start=0)

# corruption_type = 'bottom'
# cor = bottom_corruption(img_data, corrupt_row= corrupt_row, start=0)

# corruption_type = 'left'
# cor = left_corruption(img_data, corrupt_row= corrupt_row)

corruption_type = 'right'
cor = right_corruption(img_data,corrupt_row=12)


############################### Draw the results #################
ori_data = np.zeros(
    (29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8')
ori_data[0:28, :] = tile_raster_images(
        X= train_set[0][img_index],
        img_shape=(28, 28),
        tile_shape=(1, n_chains),
        tile_spacing=(1, 1))

corrupt_data = np.zeros(
    (29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8')

corrupt_data[0: 28, :] = tile_raster_images(
        X= cor,
        img_shape=(28, 28),
        tile_shape=(1, n_chains),
        tile_spacing=(1, 1))

print(corrupt_data.shape)


feed_samplor = get_samples(hidden_list=hidden_list, W=W, b=b)



feed_data = feed_samplor.get_mean_activation(input_data=cor)
print(feed_data.shape)
downact1 = reconstruct(activations=feed_data, W=W, b=b, corrupt_row=corrupt_row)    # reconstruct the images give the corruptions


if corruption_type is 'top':
    downact1[:,28*corrupt_row:] = train_set[0][img_index,:][:,28*corrupt_row:]

if corruption_type is 'bottom':
    downact1[:,: 28* (28-corrupt_row)] = train_set[0][img_index,:][:,: 28*(28-corrupt_row)]

if corruption_type is 'left':
    for i in range(28):
        downact1[:,28*i + corrupt_row: 28* (i+1)] = train_set[0][img_index,:][:,28*i + corrupt_row:28* (i+1)]

if corruption_type is 'right':
    for i in range(28):
        downact1[:,28*i:28* (i+1)-corrupt_row] = train_set[0][img_index,:][:,28*i:28* (i+1) - corrupt_row]


image_data[:28,:] = tile_raster_images(
        X= downact1,
        img_shape=(28, 28),
        tile_shape=(1, n_chains),
        tile_spacing=(1, 1)
    )

result = np.concatenate((ori_data, corrupt_data), axis = 0)
result = np.concatenate((result, image_data), axis= 0)
image = Image.fromarray(result)
image.save(savepath1 + '/ori.eps')
