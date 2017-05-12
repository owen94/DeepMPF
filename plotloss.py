import numpy as np
import matplotlib.pyplot as plt


path_1 = '../DBN_results/loss/loss_100.npy'
path_2 = '../DBN_results/loss/loss_196.npy'
path_3 = '../DBN_results/loss/loss_400.npy'

path_4 = '../DBN_results/loss/loss.eps'


loss_1 = np.load(path_1)
loss_2 = np.load(path_2)
loss_3 = np.load(path_3)

print(loss_1.shape)

x = np.arange(len(loss_1))

plt.plot(x, loss_1, 'r--')
plt.plot(x, loss_2, 'm-.')
plt.plot(x, loss_3, 'b')
plt.xlabel('Number of epoches')
plt.ylabel('Training Loss')
plt.axis([0, 400, 2, 8])
plt.legend([ 'VPF(100)', 'VPF(196)', 'VPF(400)'], loc='best')
#plt.title('Adam Training Loss')
plt.grid(False)
#plt.show()
plt.savefig(path_4)