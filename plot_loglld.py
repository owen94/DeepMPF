'''
This file deal with the log-likelihood of the VPF.
'''
import numpy as np
import matplotlib.pyplot as plt
from DBM_1 import get_samples

path1 = '/Users/liuzuozhu/MyGit/LLD/DBM_196_196_64/decay_0.001/lr_0.0001/'
path2 = '/Users/liuzuozhu/MyGit/LLD/DBM_196_196_64/decay_0.0001/lr_0.0001/'
#path3 = '/Users/liuzuozhu/MyGit/LLD/DBM_196_196_64/decay_0.0005/lr_0.001/'
path4 = '/Users/liuzuozhu/MyGit/LLD/DBM_196_196_64/decay_1e-05/lr_0.0001/'

save_path4 = '/Users/liuzuozhu/MyGit/LLD/Samples/DBM_lld.eps'


#path4 = '/Users/liuzuozhu/MyGit/Thea_mpf/hidden_1000/decay_0.0001/lr_0.001/bsz_40/'


test_lld_1 = np.load(path1 + 'test_lld.npy')
test_lld_2 = np.load(path2 + 'test_lld.npy')
#test_lld_3 = np.load(path3 + 'test_lld.npy')
test_lld_4 = np.load(path4 + 'test_lld.npy')


test_std_1 = np.load(path1 + 'test_std.npy')
test_std_2 = np.load(path2 + 'test_std.npy')

#test_std_3 = np.load(path3 + 'test_std.npy')
test_std_4 = np.load(path4 + 'test_std.npy')

# train_lld = np.load(path1 + 'train_lld.npy')
# train_std = np.load(path1 + 'train_std.npy')

x = np.arange(200, step=10)
print(np.max(np.array(test_lld_1)))
print(test_lld_1)

print(np.max(np.array(test_lld_2)))
print(test_lld_2)

# print(np.max(np.array(test_lld_3)))
# print(test_lld_3)

print(np.max(np.array(test_lld_4)))
print(test_lld_4)
#plt.errorbar(x, train_lld, train_std)
# plt.errorbar(x, test_lld_1, test_std_1,)
# plt.errorbar(x, test_lld_2, test_std_2,)
#
# #plt.errorbar(x, test_lld_3, test_std_3)
# plt.errorbar(x, test_lld_4, test_std_4,)



plt.plot(x, test_lld_1, 'r--')
plt.plot(x, test_lld_2, 'm')
plt.plot(x, test_lld_4, 'b-.')
#plt.axis([0, 200, -120, 220])
plt.xlabel('Number of epochs, learning rate = 0.0001')
plt.ylabel('Log-likelihood')
plt.legend(['Decay-0.001', 'Decay-0.0001','Decay-0.00001'], loc='best', fontsize = 14)

plt.savefig(save_path4)

# def show_lld(train, train_std, test, test_std):
#
#     x = np.arange(200, step=10)
#     plt.errorbar(x, train, train_std)
#     plt.errorbar(x, test, test_std)
#     plt.xlabel('Number of epochs')
#     plt.ylabel('Log-likelihood')
#     plt.legend(['train', 'test'], loc='best', fontsize = 14)





