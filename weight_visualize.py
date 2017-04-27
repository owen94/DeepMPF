'''
In this file, we visualize the weights filters learned in a single RBM.
At the same time, we also visualize the activations in the hidden units.
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from utils_mpf import tile_raster_images, load_mnist, sigmoid


weight1 = '../mpf_results/400/weights_499.npy'
bias1 = '../mpf_results/400/bias_499.npy'
saveName = '../mpf_results/196/196_filter.eps'

W = np.load(weight1)
b = np.load(bias1)
print(W.shape)

# f1 = np.asarray(W[:,90])
# f1 = f1.reshape((28, 28))
# print(f1.shape)
#
# image = Image.fromarray(
#         tile_raster_images( X=W.T,
#                         img_shape=(28, 28),
#                         tile_shape=(8,5),
#                         tile_spacing=(1, 1)
#                     )
#                     )
#
# image.save(saveName)
#
# filter = [0, 40, 75, 124, 182, 263,
#           90, 330, 83, 389, 225, 270,
#           12, 120, 310, 240]
#
#
# fig, axes = plt.subplots(nrows=4, ncols=4)
# cmap = matplotlib.cm.RdBu
# for i in range(16):
#     j = np.random.randint(low=0, high=W.shape[1])
#     im = axes.flat[i].imshow(np.asarray(W[:,j]).reshape((28,28)), cmap=cmap, vmin=-1, vmax=1)
#     axes.flat[i].axis('off')
#
# cax,kw = matplotlib.colorbar.make_axes([ax for ax in axes.flat])
# plt.colorbar(im, cax=cax, **kw)
# #plt.title('Filters')
# fig.savefig(saveName)



#############################  Visualize the activations / sparsity ##############
data = load_mnist()

activations = sigmoid(np.dot(data[:10000], W) + b[W.shape[0]:])

activations = np.mean(activations, axis = 0)
saveName1 = '../mpf_results/400/activations.eps'
fig1 = plt.figure()
# ax1 = fig1.add_subplot(111)
# ax1.set_title('Mean activations')
# plt.imshow( np.asarray(activations),aspect = 'auto')
# plt.colorbar()
plt.hist(activations, bins=20, color='b')
#plt.title('Mean activations of 400 hidden units')
plt.xlabel('Mean activation')
plt.ylabel('Number of hidden units')
#gplt.axis([0, 0.6, 0, 60])
fig1.savefig(saveName1)

print(np.mean(activations))