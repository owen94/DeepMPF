'''

'''

from utils_mpf import *
from PIL import Image
from DBM import *

# path1 = '../Thea_mpf/DBM/DBM_400_196_64/dbm_299.pkl'
# savepath1 = '../Thea_mpf/DBM/DBM_400_196_64'
# path2 = '../Thea_mpf/DBM/DBM_196_196_64/dbm_299.pkl'
# savepath2 = '../Thea_mpf/DBM/DBM_196_196_64'
#
# path3 = '../Thea_mpf/DBM/DBM_1000_400_64/dbm_99.pkl'
# savepath3 = '../Thea_mpf/DBM/DBM_1000_400_64'
#
# path4 = '../Thea_mpf/DBM/DBM_196_64/dbm_99.pkl'
# savepath4 = '../Thea_mpf/DBM/DBM_196_64'
#
# hidden_list = [784, 196, 64]
# dbm1 = load(path1)
# dbm2 = load(path2)
# dbm3 = load(path3)
# dbm4 = load(path4)


# path_w = '../DBM_results/Samples/196_64/weight_599.npy'
# path_b = '../DBM_results/Samples/196_64/bias_599.npy'
# savepath1 = '../DBM_results/Samples/196_64'

path_w = '../LLD/DBM_196_196_64/decay_0.0001/lr_0.001/weight_199.npy'
path_b = '../LLD/DBM_196_196_64/decay_0.0001/lr_0.001/bias_199.npy'
savepath1 = '../LLD/Samples/'

# path_w = '../LLD/hidden_100/decay_0.0001/lr_0.0001/bsz_40/weights_199.npy'
# path_b = '../LLD/hidden_100/decay_0.0001/lr_0.0001/bsz_40/bias_199.npy'


W = np.load(path_w)
b = np.load(path_b)
hidden_list = [784, 196, 196, 64]

num_rbm = len(hidden_list) -1

# W = []
# b = []
# for i in range(num_rbm):
#     W.append(dbm4.W[i].get_value(borrow = True))
#     b.append(dbm4.b[i].get_value(borrow = True))

n_chains = 8
n_samples = 8
plot_every = 5
image_data = np.zeros(
    (29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8'
)


dataset = 'mnist.pkl.gz'
f = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
f.close()

binarizer = preprocessing.Binarizer(threshold=0.5)
training_data =  binarizer.transform(train_set[0])
train_data = test_set[0]

feed_samplor = get_samples(hidden_list=hidden_list, W=W, b=b)
feed_data = feed_samplor.get_mean_activation(input_data= training_data)


#feed_data = sigmoid(np.dot(training_data,W) + b[784:])

feed_mean_activation = np.mean(feed_data, axis=0)

#seeds = []

a = np.load(savepath1 + 'seeds.npy')

for idx in range(n_samples):
    #persistent_vis_chain = np.random.binomial(n=1, p= feed_mean_activation, size=(n_chains, hidden_list[-1]))
    # persistent_vis_chain2 = np.random.binomial(n=1, p= feed_mean_activation, size=(n_chains, hidden_list[-1]))
    # persistent_vis_chain = (persistent_vis_chain + persistent_vis_chain2) % 2
    # print(persistent_vis_chain)
    #seeds += [persistent_vis_chain]

    #persistent_vis_chain = np.random.randint(2,size=(n_chains, hidden_list[-1]))
    persistent_vis_chain = np.array([ a[0][4],a[1][4], a[3][3], a[4][3], a[7][0], a[7][1] ])


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


    # W_sample = W
    # b_down = b[:784]
    # b_up = b[784:]
    # for j in range(plot_every):
    #     downact1 = sigmoid(np.dot(v_samples,W_sample.T) + b_down )
    #     down_sample1 = np.random.binomial(n=1, p= downact1)
    #     upact1 = sigmoid(np.dot(down_sample1,W_sample)+b_up)
    #     v_samples = np.random.binomial(n=1,p=upact1)
    # v_samples = down_sample1



    #
    # for j in range(plot_every):
    #     for i in range(num_rbm):
    #         vis_units = hidden_list[num_rbm-i - 1]
    #         W_sample = W[num_rbm - i -1 ][:vis_units,vis_units:]
    #         b_down = b[num_rbm - i -1 ][:vis_units]
    #         downact1 = sigmoid(np.dot(v_samples,W_sample.T) + b_down )
    #         down_sample1 = np.random.binomial(n=1, p= downact1)
    #         v_samples = down_sample1
    #     for i in range(num_rbm):
    #         vis_units = hidden_list[i]
    #         W_sample = W[i ][:vis_units,vis_units:]
    #         b_up = b[i ][vis_units:]
    #         upact1 = sigmoid(np.dot(v_samples,W_sample)+b_up)
    #         v_samples = np.random.binomial(n=1,p=upact1)
    # print(' ... plotting sample ', idx)


    image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
        X= downact1,
        img_shape=(28, 28),
        tile_shape=(1, n_chains),
        tile_spacing=(1, 1)
    )

image = Image.fromarray(image_data)
image.save(savepath1 + '/samples.eps')
#np.save(savepath1 + 'seeds.npy', seeds)
a = np.load(savepath1 + 'seeds.npy')
print(a[0])