'''
This file deal with the log-likelihood of the VPF.
'''
import numpy as np
import matplotlib.pyplot as plt

path1 = '/Users/liuzuozhu/MyGit/DBM_results/Directed_DBM/DBM_196_196_64/decay_0.01/lr_0.0001/'

test_lld = np.load(path1 + 'test_lld.npy')
test_std = np.load(path1 + 'test_std.npy')
train_lld = np.load(path1 + 'train_lld.npy')
train_std = np.load(path1 + 'train_std.npy')

x = np.arange(400, step=5)
print(test_std)
print(test_lld)
#plt.errorbar(x, train_lld, train_std)
# plt.plot(x, test_lld)
# plt.plot(x, train_lld)
#plt.show()
