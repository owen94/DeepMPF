

import numpy as np
import gzip, pickle, random
from sklearn import preprocessing
import matplotlib.pyplot as plt
from utils_mpf import  *
#from DBM import get_samples
from PIL import Image


dataset = 'mnist.pkl.gz'
f = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
f.close()
binarizer = preprocessing.Binarizer(threshold=0.5)
data =  binarizer.transform(train_set[0])

img_index = random.sample(range(0, data.shape[0]), 1)
print(img_index)