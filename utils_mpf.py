import pickle
import numpy as np
import matplotlib.pyplot as plt
import PIL, gzip
import matplotlib.colors as colors
import theano
import theano.tensor as T
plt.switch_backend('agg')
from sklearn import  preprocessing

def load(filename):
    with open(filename,'rb') as f :
        bob = pickle.load(f,encoding="bytes")
    f.close()
    return bob


def save(filename, bob):
    f = open(filename,'wb')
    pickle.dump(bob,f)
    f.close()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def normalizeData(patches):
    # Squash data to [0.1, 0.9] since we use sigmoid as the activation
    # function in the output layer

    # Remove DC (mean of images).
    patches = patches-np.array([np.mean(patches,axis=1)]).T

    # Truncate to +/-3 standard deviations and scale to -1 to 1
    pstd = 3*np.std(patches)
    patches = np.fmax(np.fmin(patches,pstd),-pstd)/pstd

    # Rescale from [-1,1] to [0.1,0.9]
    patches = (patches+1)*0.4+0.1
    return patches

def load_IMAGE():

    Images = np.load('IMAGES.npy')
    num_patches = 10000
    patch_size = 8
    patches = np.zeros((num_patches,patch_size*patch_size))

    for i in range(num_patches):
        pix_index = np.random.randint(500,size=2)
        im_index = np.random.randint(10)
        patches[i,:] = patches[i,:] + \
                       Images[pix_index[0]:pix_index[0]+patch_size,pix_index[1]:pix_index[1]+patch_size,
                       im_index].reshape(1,patch_size*patch_size)


    patches = normalizeData(patches)
    #display(patches[:100,:])

    binarizer = preprocessing.Binarizer(threshold=0.5)
    patches =  binarizer.transform(patches)

    return patches


def load_mnist():
    dataset = 'mnist.pkl.gz'
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
    f.close()

    binarizer = preprocessing.Binarizer(threshold=0.5)
    data =  binarizer.transform(train_set[0])

    return data

def get_new_data(data,W,b):

    activation = sigmoid(np.dot(data,W) + b)
    hidden_samples = np.random.binomial(n=1, p = activation)
    new_input = np.concatenate((data,hidden_samples), axis = 1)

    return activation, new_input


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    if dataset == 'mnist.pkl.gz':
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
        f.close()

    else:
        f = open(dataset,'rb')
        train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
        f.close()

    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def get_mpf_params(visible_units, hidden_units):
    '''
    :param visible_units: number of units in the visible layer
    :param hidden_units: number of units ni the hidden layer
    :return: The well structured MPF weight matrix
    The MPF weight matrix is of the form:
    [0,   W,
     W.T, 0]
    '''
    numpy_rng = np.random.RandomState(555555)
    W = numpy_rng.randn(visible_units,hidden_units)/np.sqrt(visible_units*hidden_units)
    W_up = np.concatenate((np.zeros((visible_units,visible_units)), W), axis = 1)
    W_down = np.concatenate((W.T,np.zeros((hidden_units,hidden_units))), axis = 1 )

    W = np.concatenate((W_up,W_down), axis = 0)

    print(W.shape)

    return W


def show_loss(savename, epoch_error = None):

    x = np.arange(len(epoch_error))
    plt.plot(x, epoch_error, 'r')
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.title('Adam Training Loss')
    plt.grid(True)
    plt.savefig(savename)


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.
    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).
    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.
    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image
    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)
    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats
    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not
    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.
    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array



def displayNetwork(A,optNormEach=False,optNormAll=True,numColumns=None,imageWidth=None,cmapName='gray',
                   borderColor='black',borderWidth=1,verbose=True,graphicsLibrary='matplotlib',saveName=''):
    # This function visualizes filters in matrix A. Each row of A is a
    # filter. We will reshape each row into a square image and visualizes
    # on each cell of the visualization panel. All other parameters are
    # optional, usually you do not need to worry about them.
    #
    # optNormEach: whether we need to normalize each row so that
    # the mean of each row is zero.
    #
    # optNormAll: whether we need to normalize all the rows so that
    # the mean of all the rows together is zero.
    #
    # imageWidth: how many pixels are there for each image
    # Default value is the squareroot of the number of columns in A.
    #
    # numColumns: how many columns are there in the display.
    # Default value is the squareroot of the number of rows in A.

    # compute number of rows and columns
    nr,nc = np.shape(A)
    if imageWidth==None:
        sx = np.ceil(np.sqrt(nc))
        sy = np.ceil(nc/sx)
    else:
        sx = imageWidth
        sy = np.ceil(nc/sx)
    if numColumns==None:
        n = np.ceil(np.sqrt(nr))
        m = np.ceil(nr/n)
    else:
        n = numColumns
        m = np.ceil(nr/n)
    n = np.uint8(n)
    m = np.uint8(m)
    if optNormAll:
        A = A-A.min()
        A = A/A.max()

    # insert data onto squares on the screen
    k = 0
    buf = borderWidth
    array = -np.ones([buf+m*(sy+buf),buf+n*(sx+buf)])
    for i in range(1,m+1):
        for j in range(1,n+1):
            if k>=nr:
                continue
            B = A[k,:]
            if optNormEach:
                B = B-B.min()
                B = B/float(B.max())
            B = np.reshape(np.concatenate((B,-np.ones(sx*sy-nc)),axis=0),(sx,-1))
            array[(buf+(i-1)*(sy+buf)):(i*(sy+buf)),(buf+(j-1)*(sx+buf)):(j*(sx+buf))] = B
            k = k+1

    # display picture and save it
    cmap = plt.cm.get_cmap(cmapName)
    cmap.set_under(color=borderColor)
    cmap.set_bad(color=borderColor)
    if graphicsLibrary=='PIL':
        im = PIL.Image.fromarray(np.uint8(cmap(array)*255))
        if verbose:
            im.show()
        if saveName != '':
            im.save(saveName)
    elif graphicsLibrary=='matplotlib':
        plt.imshow(array,interpolation='nearest',
                   norm=colors.Normalize(vmin=0.0,vmax=1.0,clip=False))
        plt.set_cmap(cmap)
        # if verbose:
        #     #plt.show()
        if saveName != '':
            plt.savefig(saveName)
