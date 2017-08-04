'''
This file deal with the log-likelihood of the VPF.
'''
import numpy as np
import matplotlib.pyplot as plt

path1 = '/Users/liuzuozhu/MyGit/DBM_results/Directed_DBM/DBM_196_196_64/decay_0.01/a_single_lld_lr_0.0001/'

test_lld = np.load(path1 + 'test_lld.npy')
test_std = np.load(path1 + 'test_std.npy')
train_lld = np.load(path1 + 'train_lld.npy')
train_std = np.load(path1 + 'train_std.npy')

x = np.arange(400, step=5)
print(train_std)
print(train_lld)
plt.errorbar(x, train_lld, train_std)
plt.errorbar(x, test_lld, test_std)

# plt.plot(x, test_lld)
# plt.plot(x, train_lld)
plt.xlabel('Number of epochs')
plt.ylabel('Log-likelihood')
plt.legend(['test', 'train'], loc='best', fontsize = 14)

plt.show()


def show_lld(train, train_std, test, test_std):

    x = np.arange(200, step=10)
    plt.errorbar(x, train, train_std)
    plt.errorbar(x, test, test_std)
    plt.xlabel('Number of epochs')
    plt.ylabel('Log-likelihood')
    plt.legend(['train', 'test'], loc='best', fontsize = 14)





