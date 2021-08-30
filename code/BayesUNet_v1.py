#!/usr/bin/env python
# coding: utf-8


from __future__ import absolute_import

from functools import partial
import numpy as np
import cv2
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
import json
import warnings
from math import ceil, floor

import tensorflow as tf
import tensorflow.keras 
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import ZeroPadding3D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.layers import Cropping2D, Conv2DTranspose
from tensorflow.keras.layers import Lambda
from tensorflow.keras import initializers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Reshape
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# layers/mc_dropout.py

class MCDropout(Dropout):
    """ Drops elements of input variable randomly.
    See the paper by Y. Gal, and G. Zoubin: `Dropout as a bayesian approximation: \
    Representing model uncertainty in deep learning .\
    <https://arxiv.org/abs/1506.02142>`
    """

    def call(self, inputs, training=None):

        if training is not None:
            if not training:
                warnings.warn('Training option is ignored..')

        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=True) # NOTE: force
        return inputs



# initializers.py

def _kernel_center(ksize):

    center = [None] * len(ksize)
    factor = [None] * len(ksize)

    for i, s in enumerate(ksize):

        factor[i] = (s + 1) // 2
        if s % 2 == 1:
            center[i] = factor[i] - 1
        else:
            center[i] = factor[i] - 0.5

    return center, factor


def _bilinear_kernel_2d(ksize):
    """ Get a kernel upsampling by bilinear interpolation
    Args:
        ksize (list of int): Kernel size.
    Returns:
        numpy.ndarray: A kernel.
    See also:
        https://arxiv.org/pdf/1411.4038.pdf
        https://github.com/d2l-ai/d2l-en/blob/master/chapter_computer-vision/fcn.md#initialize-the-transposed-convolution-layer
    """

    assert len(ksize) == 2

    og = np.ogrid[:ksize[0], :ksize[1]]
    center, factor = _kernel_center(ksize)

    kernel = (1 - abs(og[0] - center[0]) / factor[0]) *              (1 - abs(og[1] - center[1]) / factor[1])

    return kernel


def _bilinear_kernel_3d(ksize):

    assert len(ksize) == 3

    og = np.ogrid[:ksize[0], :ksize[1], :ksize[2]]
    center, factor = _kernel_center(ksize)

    kernel = (1 - abs(og[0] - center[0]) / factor[0]) *              (1 - abs(og[1] - center[1]) / factor[1]) *              (1 - abs(og[2] - center[2]) / factor[2])

    return kernel


def _bilinear_kernel_nd(ksize, dtype=np.float32):

    if len(ksize) == 2:
        kernel = _bilinear_kernel_2d(ksize)
    elif len(ksize) == 3:
        kernel = _bilinear_kernel_3d(ksize)
    else:
        raise NotImplementedError()

    return kernel.astype(dtype)


class BilinearUpsample(Initializer):
    """ Initializer of Bilinear upsampling kernel for convolutional weights.
    See also: https://arxiv.org/pdf/1411.4038.pdf
    """
    def __init__(self, scale=1.):
        super().__init__()
        self.scale = scale

    def __call__(self, shape, dtype=None):
        shape = shape[::-1] # NOTE: transpose

        ndim = len(shape)
        in_channels, out_channels = shape[:2]
        ksize = shape[2:]
        kernel = self.scale * _bilinear_kernel_nd(ksize)

        weight = np.zeros(shape)
        weight[range(in_channels),range(out_channels),...] = kernel
        weight = weight.transpose(np.arange(ndim)[::-1]) # NOTE: transpose

        if dtype is None:
            dtype = K.floatx()

        return tf.convert_to_tensor(weight, dtype=dtype)


def bilinear_upsample(scale=1.):
    return BilinearUpsample(scale)



# models/__init__.py

class ModelArchitect(metaclass=ABCMeta):
    """ Base class of model architecture
    """

    def save_args(self, out, args):
        args = args.copy()
        ignore_keys = ['__class__', 'self']
        for key in ignore_keys:
            if key in args.keys():
                args.pop(key)

        with open(out, 'w', encoding='utf-8') as f:
            json.dump(args, f, ensure_ascii=False, indent=4)

    @abstractmethod
    def build(self):
        raise NotImplementedError()



# models/mc_sampler.py 

_batch_axis = 0
_channel_axis = -1

_reduce_table = {
    'none': lambda x: x,
    'mean': partial(K.mean, axis=_channel_axis),
    'std': partial(K.std, axis=_channel_axis),
    'var': partial(K.var, axis=_channel_axis),
    'argmax': partial(K.argmax, axis=_channel_axis),
    'argmin': partial(K.argmin, axis=_channel_axis),
}

class MCSampler(ModelArchitect):
    """ Monte Carlo estimation to approximate the predictive distribution.
    Predictive variance is a metric indicating uncertainty.
    Args:
        predictor (~keras.models.Model): Predictor network.
        mc_iteration (int): Number of iterations in MCMC sampling
        activation (str, optional): Activation function.
            Defaults to 'softmax'.
        reduce_mean (str, optional): Reduce function along the channel axis for mean tensor.
            Defaults to 'argmax'.
        reduce_var (str, optional): Reduce function along the channel axis for variance tensor.
            Defaults to 'mean'.
    Note:
        Default values ​​are set assuming segmentation task.
    See also: https://arxiv.org/pdf/1506.02142.pdf
              https://arxiv.org/pdf/1511.02680.pdf
    """
    def __init__(self,
                 predictor,
                 mc_iteration,
                 activation='softmax',
                 reduce_mean='argmax',
                 reduce_var='mean'):

        self._predictor = predictor
        self._mc_iteration = mc_iteration
        self._activation = activation
        self._reduce_mean = reduce_mean
        self._reduce_var = reduce_var

#        input_shape = predictor.layers[0].input_shape[1:]
        input_shape = predictor.layers[0].input.shape[1:]
        self._input_shape = input_shape


    @property
    def input_shape(self):
        return self._input_shape

    @property
    def predictor(self):
        return self._predictor
    @property
    def mc_iteration(self):
        return self._mc_iteration

    @property
    def activation(self):
        if self._activation is not None:
            return Activation(self._activation)
        else:
            return lambda x: x

    @property
    def reduce_mean(self):
        return _reduce_table[self._reduce_mean]

    @property
    def reduce_var(self):
        return _reduce_table[self._reduce_var]

    def build(self):

        inputs = Input(self.input_shape)

        mc_samples = Lambda(lambda x: K.repeat_elements(x, self.mc_iteration, axis=_batch_axis))(inputs)

        probs = self.predictor(mc_samples)

        ret_shape = self.predictor.layers[-1].output_shape
        ret_shape = (-1, self.mc_iteration, *ret_shape[1:])

        probs = Lambda(lambda x: K.reshape(x, ret_shape))(probs)

        mean = Lambda(lambda x: K.mean(x, axis=1))(probs)
        mean = Lambda(lambda x: self.reduce_mean(x))(mean)

        variance = Lambda(lambda x: K.var(x, axis=1))(probs)
        variance = Lambda(lambda x: self.reduce_var(x))(variance)

        return Model(inputs=inputs, outputs=[mean, variance])
        

# models/unet_2d.py
_batch_axis = 0
_row_axis = 1
_col_axis = 2
_channel_axis = -1


class UNet2D(ModelArchitect):
    """ Two-dimensional U-Net

    Args:
        input_shape (list): Shape of an input tensor.
        out_channels (int): Number of output channels.
        nlayer (int, optional): Number of layers.
            Defaults to 5.
        nfilter (list or int, optional): Number of filters.
            Defaults to 32.
        ninner (list or int, optional): Number of layers in UNetBlock.
            Defaults to 2.
        kernel_size (list or int): Size of convolutional kernel. Defaults to 3.
        activation (str, optional): Type of activation layer.
            Defaults to 'relu'.
        conv_init (str, optional): Type of kernel initializer for conv. layer.
            Defaults to 'he_normal'.
        upconv_init (str, optional): Type of kernel initializer for upconv. layer.
            Defaults to 'he_normal'.
        bias_init (str, optional): Type of bias initializer for conv. and upconv. layer.
        dropout (bool, optional): If True, enables the dropout.
            Defaults to False.
        drop_prob (float, optional): Ratio of dropout.
            Defaults to 0.5.
        pool_size (int, optional): Size of spatial pooling.
            Defaults to 2.
        batch_norm (bool, optional): If True, enables the batch normalization.
            Defaults to False.
    """
    def __init__(self,
                 input_shape,
                 out_channels,
                 nlayer=5,
                 nfilter=32,
                 ninner=2,
                 kernel_size=3,
                 activation='relu',
                 conv_init='he_normal',
                 upconv_init='he_normal',
                 bias_init='zeros',
                 dropout=True,
                 drop_prob=0.5,
                 pool_size=2,
                 batch_norm=False):

        super(ModelArchitect, self).__init__()

        self._args = locals()

        self._input_shape = input_shape
        self._out_channels = out_channels
        self._nlayer = nlayer
        self._nfilter = nfilter
        self._ninner = ninner
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self._kernel_size = kernel_size
        self._activation = activation
        self._conv_init = conv_init
        self._upconv_init = upconv_init
        self._bias_init = bias_init
        self._dropout = dropout
        self._drop_prob = drop_prob
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        self._pool_size = pool_size
        self._batch_norm = batch_norm

    def save_args(self, out):
        super().save_args(out, self._args)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def ninner(self):
        return self._ninner

    @property
    def nlayer(self):
        return self._nlayer

    @property
    def nfilter(self):
        return self._nfilter

    @property
    def pool_size(self):
        return self._pool_size

    @property
    def activation(self):
        return Activation(self._activation)

    @property
    def kernel_size(self):
        return self._kernel_size

    @property
    def conv_init(self):
        return initializers.get(self._conv_init)

    @property
    def upconv_init(self):

        if self._upconv_init == 'bilinear':
            return bilinear_upsample()

        return initializers.get(self._upconv_init)

    @property
    def bias_init(self):
        return self._bias_init

    @property
    def pad(self):
        return ZeroPadding2D #ReflectionPadding2D

    @property
    def conv(self):
        return partial(Conv2D,
                        padding='valid', use_bias=True,
                        kernel_initializer=self.conv_init,
                        bias_initializer=self.bias_init)

    @property
    def upconv(self):

        return partial(Conv2DTranspose,
                        strides=self.pool_size,
                        padding='valid', use_bias=True,
                        kernel_initializer=self.upconv_init,
                        bias_initializer=self.bias_init)
    @property
    def pool(self):
        return MaxPooling2D(pool_size=self.pool_size)


    @property
    def dropout(self):
        if self._dropout:
            return partial(Dropout(self._drop_prob))
        else:
            return lambda x: x

    @property
    def norm(self):
        if self._batch_norm:
            return BatchNormalization(axis=_channel_axis)
        else:
            return lambda x: x

    def concat(self, x1, x2):

        dx = (x1.shape[_row_axis] - x2.shape[_row_axis]) / 2
        dy = (x1.shape[_col_axis] - x2.shape[_col_axis]) / 2

        crop_size = ((floor(dx), ceil(dx)), (floor(dy), ceil(dy)))

        x12 = Concatenate(axis=_channel_axis)([Cropping2D(crop_size)(x1), x2])

        return x12

    def base_block(self, x, nfilter, ksize):

        h = x

        for _ in range(self.ninner):
            h = self.pad([(k-1)//2 for k in ksize])(h)
            h = self.conv(nfilter, ksize)(h)
            h = self.norm(h)
            h = self.activation(h)

        h = self.dropout(h)

        return h

    def contraction_block(self, x, nfilter):

        h = self.base_block(x, nfilter, self.kernel_size)
        o = self.pool(h)

        return h, o

    def expansion_block(self, x1, x2, nfilter):

        ksize = self.kernel_size

        h1 = self.pad([(k-1)//2 for k in ksize])(x1)
        h1 = self.upconv(x1.shape[_channel_axis], ksize)(h1)
        h1 = self.norm(h1)
        h1 = self.activation(h1)

        h = self.concat(h1, x2)
        h = self.base_block(h, nfilter, ksize)

        return h

    def build(self):
        inputs = Input(self.input_shape)

        store_activations = []

        # down
        h_pool = inputs
        for i in range(self.nlayer):
            nfilter = self.nfilter * (2 ** (i))
            h_conv, h_pool = self.contraction_block(h_pool, nfilter)
            store_activations.append(h_conv)

        store_activations = store_activations[::-1] # NOTE: reversed

        # up
        h = store_activations[0]
        for i in range(1, self.nlayer):
            nfilter = self.nfilter * (2 ** (self.nlayer-i-1))
            h = self.expansion_block(h, store_activations[i], nfilter)

        # out
        h = self.pad(1)(h)
#        outputs = self.conv(self.out_channels, 3)(h)

        h = self.conv(self.out_channels, 3)(h)
        outputs = Activation('softmax')(h)

        return Model(inputs=inputs, outputs=outputs)



# models/bayesian_unet_2d.py
class BayesianUNet2D(UNet2D):
    """ Two-dimensional Bayesian U-Net

    Args:
        input_shape (list): Shape of an input tensor.
        out_channels (int): Number of output channels.
        nlayer (int, optional): Number of layers.
            Defaults to 5.
        nfilter (list or int, optional): Number of filters.
            Defaults to 32.
        ninner (list or int, optional): Number of layers in UNetBlock.
            Defaults to 2.
        kernel_size (list or int): Size of convolutional kernel. Defaults to 3.
        activation (str, optional): Type of activation layer.
            Defaults to 'relu'.
        conv_init (str, optional): Type of kernel initializer for conv. layer.
            Defaults to 'he_normal'.
        upconv_init (str, optional): Type of kernel initializer for upconv. layer.
            Defaults to 'he_normal'.
        bias_init (str, optional): Type of bias initializer for conv. and upconv. layer.
        drop_prob (float, optional): Ratio of dropout.
            Defaults to 0.5.
        pool_size (int, optional): Size of spatial pooling.
            Defaults to 2.
        batch_norm (bool, optional): If True, enables the batch normalization.
            Defaults to False.
    """
    def __init__(self,
                 input_shape,
                 out_channels,
                 nlayer=5,
                 nfilter=32,
                 ninner=2,
                 kernel_size=3,
                 activation='relu',
                 conv_init='he_normal',
                 upconv_init='he_normal',
                 bias_init='zeros',
                 drop_prob=0.5,
                 pool_size=2,
                 batch_norm=False):

        args = locals()
        ignore_keys = ['__class__', 'self']
        for key in ignore_keys:
            if key in args.keys():
                args.pop(key)
        args['dropout'] = True # NOTE: force
        super().__init__(**args)

    @property
    def dropout(self):
        if self._dropout:
            return partial(MCDropout(self._drop_prob))
        else:
            raise ValueError()


def iou(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def get_augmented(
    X_train, 
    Y_train, 
    X_val=None,
    Y_val=None,
    batch_size=32, 
    seed=0, 
    data_gen_args = dict(
        rotation_range=10.,
        width_shift_range=0.02,
        height_shift_range=0.02,
        zca_whitening = False,
        zca_epsilon = 1e-6,
        shear_range=5,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest'
    )):


    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y1_datagen = ImageDataGenerator(**data_gen_args)
    Y2_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y1_datagen.fit(Y_train[:,:,:,0:1], augment=True, seed=seed)
    Y2_datagen.fit(Y_train[:,:,:,1:2], augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
    Y1_train_augmented = Y1_datagen.flow(Y_train[:,:,:,0:1], batch_size=batch_size, shuffle=True, seed=seed)
    Y2_train_augmented = Y2_datagen.flow(Y_train[:,:,:,1:2], batch_size=batch_size, shuffle=True, seed=seed)
    
    train_generator = zip(X_train_augmented, Y1_train_augmented,Y2_train_augmented)
    return train_generator

def my_generator(
    X_train, 
    Y_train,
    train_gen,
    X_val=None,
    Y_val=None,
    batch_size=2, 
    seed=0, 
    data_gen_args = dict(
        rotation_range=10.,
        width_shift_range=0.02,
        height_shift_range=0.02,
        zca_whitening = False,
        zca_epsilon = 1e-6,
        shear_range=5,
        zoom_range=0.3,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode='nearest'
    )):
    while 1:
        sample_batch = next(train_gen)
        xx, yy1,yy2 = sample_batch
        yy = np.zeros((xx.shape[0],xx.shape[1],xx.shape[2],2),dtype=np.float32)
        yy[:,:,:,0:1] = yy1
        yy[:,:,:,1:2] = yy2
        yield (xx, yy)


# Read data

from PIL import Image
import matplotlib.pyplot as plt

import glob

masks = glob.glob("./Dane/Labels/mask*.bmp")
orgs = glob.glob("./Dane/Images/org*.bmp")
masks.sort()
orgs.sort()

print(len(masks))
print(len(orgs))

imgs_list = []
masks_list = []
for image, mask in zip(orgs, masks):
    im = np.array(Image.open(image))
    im = im[:,:,0]
    imgs_list.append(im)
    im = np.array(Image.open(mask))
    im = im[:,:,0:2]
    im[im!=0] = 1
    im[:,:,1] = 1-im[:,:,1]
    masks_list.append(im)
    
imgs_np = np.asarray(imgs_list)
masks_np = np.asarray(masks_list)

#plt.plot()
#plt.imshow(imgs_np[0,:,:],cmap='gray')
#plt.show()

#plt.plot()
#plt.imshow(masks_np[0,:,:,0],cmap='gray')
#plt.show()

print(imgs_np.max(), masks_np.max())
x = np.asarray(imgs_np, dtype=np.float32)/255
y = np.asarray(masks_np, dtype=np.float32)
print(x.max(), y.max())
print(x.shape, y.shape)
y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 2)
print(x.shape, y.shape)
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
print(x.shape, y.shape)



from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_val: ", x_val.shape)
print("y_val: ", y_val.shape)



data_gen_args = dict(
        rotation_range=10.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
)
  
train_gen = get_augmented(x_train, y_train, batch_size=2,data_gen_args=data_gen_args)
generator = my_generator(x_train, y_train,train_gen, batch_size=2,data_gen_args=data_gen_args)



input_shape = (128,128,1)

archit = BayesianUNet2D(input_shape, 2, nlayer=3)
model = archit.build()
#model.summary()

model.compile(
    optimizer=tensorflow.keras.optimizers.Adam(lr=0.0001), 
    loss = 'categorical_crossentropy',
    metrics=[iou]
)

model_filename = 'segm_model.h5'
callback_checkpoint = ModelCheckpoint(
    model_filename, 
    verbose=1, 
    monitor='val_loss', 
    save_best_only=True
)


# In[ ]:



history = model.fit(generator,steps_per_epoch=100,epochs=10,validation_data=(x_val, y_val),callbacks=[callback_checkpoint])


# In[ ]:


"""
samples = np.asarray([x[0] for _ in range(10)])
outputs = model(samples)
print(samples.shape,outputs.shape)

mean = np.mean(outputs,axis=0)
std = np.std(outputs,axis=0)

plt.plot()
plt.imshow(mean[:,:,0],cmap='gray')
plt.show()    

plt.plot()
plt.imshow(std[:,:,0],cmap='gray')
plt.show()    

plt.plot()
plt.imshow(std[:,:,1],cmap='gray')
plt.show()    

for i in range(10):
    plt.plot()
    plt.imshow(outputs[i,:,:,0],cmap='gray')
    plt.show()
"""


# In[ ]:


#predictor = MCSampler(model, mc_iteration=10).build()
#outputsMC = predictor(x[0:1])
#print(outputsMC[0].shape)

#plt.plot()
#plt.imshow(outputsMC[0][0,:,:],cmap='gray')
#plt.show()    

#plt.plot()
#plt.imshow(outputsMC[1][0,:,:],cmap='gray')
#plt.show() 

