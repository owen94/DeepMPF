'''

'''

from utils_mpf import *
from PIL import Image
from DBM import *

path1 = '../Thea_mpf/DBM/DBM_400_196_64/dbm_299.pkl'
savepath1 = '../Thea_mpf/DBM/DBM_400_196_64'
path2 = '../Thea_mpf/DBM/DBM_196_196_64/dbm_299.pkl'
savepath2 = '../Thea_mpf/DBM/DBM_196_196_64'

path3 = '../Thea_mpf/DBM/DBM_1000_400_64/dbm_99.pkl'
savepath3 = '../Thea_mpf/DBM/DBM_1000_400_64'



hidden_list = [784, 1000, 400, 64]
dbm1 = load(path1)
dbm2 = load(path2)
dbm3 = load(path3)
num_rbm = 3

W = []
b = []
for i in range(num_rbm):
    W.append(dbm3.W[i].get_value(borrow = True))
    b.append(dbm3.b[i].get_value(borrow = True))

n_chains = 20
n_samples = 10
plot_every = 4
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
image.save(savepath3 + '/samples.png')