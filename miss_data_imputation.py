'''
Give a trained model, this will test the imputation performance..
'''

import numpy as np
import gzip, pickle, random
from sklearn import preprocessing
import matplotlib.pyplot as plt



dataset = 'mnist.pkl.gz'
f = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
f.close()
binarizer = preprocessing.Binarizer(threshold=0.5)
data =  binarizer.transform(train_set[0])

img_index = random.sample((0,data.shape[1]), 20)
img_data = data[img_index,:]




def corruption():
    # corrupt the images





def reconstruct():
    # reconstruct the images give the corruptions


