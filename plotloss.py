import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#
path_1 = '../DBN_results/loss/loss_100.npy'
path_2 = '../DBN_results/loss/loss_196.npy'
path_3 = '../DBN_results/loss/loss_400.npy'

path_4 = '../DBN_results/loss/loss.eps'

#
loss_1 = np.load(path_1)
loss_2 = np.load(path_2)
loss_3 = np.load(path_3)

print(loss_1.shape)

x = np.arange(len(loss_1))

plt.plot(x, loss_1, 'r--')
plt.plot(x, loss_2, 'm-.')
plt.plot(x, loss_3, 'b')
plt.xlabel('Number of epoches', fontsize = 14)
plt.ylabel('Training Loss', fontsize = 14)
plt.axis([0, 400, 2, 8])
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.legend([ 'RBM(100)', 'RBM(196)', 'RBM(400)'], loc='best', fontsize = 14)
#plt.title('Adam Training Loss')
plt.grid(False)
#plt.show()
plt.savefig(path_4)


####################################plot the sparsity     ############################
# path_4 = '../DBN_results/sparsity/squared_weight_400.npy'
# path_5 = '../DBN_results/sparsity/sparsity_params_400.npy'
#
# squaredw = np.load(path_4)
# sparsity = np.load(path_5)
#
# print(squaredw.shape)
# print(sparsity.shape)

# x = np.arange(len(squaredw))
#
# plt.plot(x, squaredw, 'r--')
# plt.plot(x, sparsity, 'm')
# plt.xlabel('Number of epoches')
# plt.ylabel('Training Loss')
# plt.axis([0, 400, 2, 8])
# plt.legend([ 'VPF(100)', 'VPF(196)', 'VPF(400)'], loc='best')
# #plt.title('Adam Training Loss')
# plt.grid(False)
# plt.show()
# plt.savefig(path_4)

# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True
#
# ticks = [0, 20, 50, 100, 200, 300, 400]
# #list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#
# fig, ax1 = plt.subplots()
# lns1 = ax1.plot(x, sparsity, 'b--', label= 'Weight sparsity')
# ax1.set_xlabel('Number of epoches', fontsize = 16)
# # Make the y-axis label, ticks and tick labels match the line color.
# ax1.set_ylabel('Weight sparsity $\\rho$', color='b', fontsize=16)
# plt.xticks(ticks, fontsize = 14)
# #yticks = [0.1,0.15,  0.2,0.25, 0.3, 0.35, 0.4, 0.45]
# yticks = [0.1,0.15, 0.2,0.25, 0.3, 0.35]
# plt.yticks(yticks,fontsize = 14)
# plt.axis([0, 400, 0.1, 0.35])
# #plt.legend([ 'Weight sparsity'])
# ax1.tick_params('y', colors='b')
#
#
# ax2 = ax1.twinx()
# lns2 = ax2.plot(x, squaredw, 'r', label= 'Squared weights')
# plt.axis([0, 400, 0, 14 ])
# plt.yticks(fontsize=14)
# ax2.set_ylabel('Squared weights $W^2$', color='r', fontsize=16)
# ax2.tick_params('y', colors='r')
# #plt.legend([ 'Squared weights'])
#
#
# lns = lns1+lns2
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc=3, fontsize = 16)
#
# saveName_sparsity = '../DBN_results/sparsity/weight_sparsity.eps'
# fig.tight_layout()
# #plt.show()
# plt.savefig(saveName_sparsity)


