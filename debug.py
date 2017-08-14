
import numpy as np

path1 = '/Users/liuzuozhu/MyGit/LLD/Directed_DBM/DBM_196_196_64/decay_1e-05/lr_0.001/test_lld.npy'

path2 = '/Users/liuzuozhu/MyGit/LLD/Directed_DBM/DBM_400_196_64/decay_0.0001/lr_0.001/test_lld_1.npy'

path3 = '/Users/liuzuozhu/MyGit/LLD/Directed_DBM/DBM_400_400_100/decay_0.0001/lr_0.001/test_lld_1.npy'
path4 = '/Users/liuzuozhu/MyGit/LLD/Directed_DBM/DBM_400_400_196/decay_0.0001/lr_0.001/test_lld_1.npy'
path5 = '/Users/liuzuozhu/MyGit/LLD/Directed_DBM/DBM_400_196_100/decay_0.0001/lr_0.001/test_lld_1.npy'

a1 = np.max(np.load(path1))
a2 = np.max(np.load(path2))
a3 = np.max(np.load(path3))
a4 = np.max(np.load(path4))
a5 = np.max(np.load(path5))
print(a1, a2, a3, a4, a5)