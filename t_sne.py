import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, preprocessing
from KL_mpf import load_mnist, sigmoid
import gzip, pickle, time


dataset = 'mnist.pkl.gz'
f = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
f.close()
binarizer = preprocessing.Binarizer(threshold=0.5)
data =  binarizer.transform(train_set[0][:20000])
targets = train_set[1]
print(data.shape)
print(targets.shape)

def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    for i in range(1000):
        plt.text(X[i, 0], X[i, 1], str(targets[i]),
                 color=plt.cm.Set1(targets[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})



def t_sne(weight,bias,weight2, bias2,perplexity,savename):

    visible_units = data.shape[1]
    W1 = np.load(weight)
    #W = W[:visible_units,visible_units:]
    b1 = np.load(bias)
    b1 = b1[visible_units:]

    W2 = np.load(weight2)
    b2 = np.load(bias2)
    b2 = b2[W2.shape[0]:]

    activation1 = sigmoid(np.dot(data,W1) + b1.reshape([1,-1]))
    activation2 = sigmoid(np.dot(activation1,W2) + b2.reshape([1,-1]))

    print(activation2.shape)
    print(activation2[1])

    tsne = manifold.TSNE(n_components=2, perplexity= perplexity, early_exaggeration=4.0, learning_rate=1000.0,
                         n_iter = 2000, init='pca')

    X_tsne = tsne.fit_transform(activation2)

    print(X_tsne.shape)

    #t0 = time()

    plot_embedding(X_tsne,
               "t-SNE embedding of the digits")
    plt.savefig(savename)

    plt.show()

    return X_tsne

def search_tsne():
    perplexity = [10]

    # weight = '../mpf_results/40/weights_499.npy'
    # bias = '../mpf_results/40/bias_499.npy'
    # hidden_units = 40
    #
    # for i in perplexity:
    #     path = '../mpf_results/40/tsne_perp_' + str(i) + '.png'
    #     t_sne(weight=weight,bias=bias,hidden_units=hidden_units, perplexity=i, savename=path)

    decay_list = [ 0.0001]
    lr_list = [0.001,0.01,0.00001]
    weight1 = '../mpf_results/lay1_196/weights_499.npy'
    bias1 = '../mpf_results/lay1_196/bias_499.npy'
    perplexity = 20

    for decy in decay_list:
        for lr in lr_list:
            weight2 = '../mpf_results/lay1_196/lay2_40/decay_' + str(decy) + '/lr_' + str(lr) + '/weights_499.npy'
            bias2 = '../mpf_results/lay1_196/lay2_40/decay_' + str(decy) + '/lr_' + str(lr) + '/bias_499.npy'
            path = '../mpf_results/lay1_196/tsne_perp.png'
            t_sne(weight=weight1,bias=bias1,weight2=weight2,bias2=bias2, perplexity=20, savename=path)

search_tsne()