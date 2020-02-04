"""WGAN-GP ResNet for CIFAR-10"""

import os, sys
sys.path.append(os.getcwd())
import shutil
import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.save_images
import tflib.cifar10
import tflib.inception_score
import tflib.plot

import numpy as np
import tensorflow as tf
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import time
import functools
import locale
locale.setlocale(locale.LC_ALL, '')

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = './cifar-10-batches-py'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

N_GPUS = 1
if N_GPUS not in [1,2]:
    raise Exception('Only 1 or 2 GPUs supported!')

BATCH_SIZE = 64 # Critic batch size
GEN_BS_MULTIPLE = 2 # Generator batch size, as a multiple of BATCH_SIZE
ITERS = 10000 # How many iterations to train for
DIM_G = 128 # Generator dimensionality
DIM_D = 64 # Critic dimensionality
NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = False # Use batchnorm (or layernorm) in critic?
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (32*32*3)
LR = 1e-4 # Initial learning rate #改成了2e-5,原本是2e-4#改成了5e-5#又改回2e-4#由改到1e-4
DECAY = True # Whether to decay LR over learning
N_CRITIC = 5 # Critic steps per generator steps
INCEPTION_FREQUENCY = 8000 # How frequently to calculate Inception score

CONDITIONAL = False # Whether to train a conditional or unconditional model
ACGAN = True # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.2 # How to scale generator's ACGAN loss relative to WGAN loss

LOAD_MARK = True
G_SYMBOL = False
D_SYMBOL = False

ST_ITERS = 2000

minus_value = 0.2 #
T_value = 1 - minus_value
start_point = 0
model_number = 5999
ITERS = ST_ITERS

if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
    print("WARNING! Conditional model without normalization in D might be effectively unconditional!")

DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]
if len(DEVICES) == 1: # Hack because the code assumes 2 GPUs
    DEVICES = [DEVICES[0], DEVICES[0]]

lib.print_model_settings(locals().copy())

def nonlinearity(x):
    return tf.nn.relu(x)
#加了load_offset,load_scale,load_moving_mean,load_moving_variance,load_mark
def Normalize(name, inputs,load_offset,load_scale,load_moving_mean,load_moving_variance,load_mark=False,labels=None):
    """This is messy, but basically it chooses between batchnorm, layernorm,
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""
    if not CONDITIONAL:
        labels = None
    if CONDITIONAL and ACGAN and ('Discriminator' in name):
        labels = None

    if ('Discriminator' in name) and NORMALIZATION_D:
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs,labels=labels,n_labels=10)
    elif ('Generator' in name) and NORMALIZATION_G:
        if labels is not None:
           return lib.ops.cond_batchnorm.Batchnorm(name, [0, 2, 3], inputs, load_offset, load_scale, load_mark=load_mark,
                                                        labels=labels, n_labels=10)
           #return lib.ops.cond_batchnorm.Batchnorm(name,[0,2,3],inputs,labels=labels,n_labels=10)
        else:
            return lib.ops.batchnorm.Batchnorm(name, [0,2,3], inputs, load_offset, load_scale, load_moving_mean,
                                               load_moving_variance, load_mark=load_mark, fused=True)
    else:
        return inputs

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, Load_filters, Load_biases, Load_Symbol, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, Load_filters, Load_biases ,
                                   Load_Symbol=Load_Symbol, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, Load_filters, Load_biases, Load_Symbol, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, Load_filters, Load_biases ,
                                   Load_Symbol=Load_Symbol, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, Load_filters, Load_biases, Load_Symbol, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, Load_filters, Load_biases,
                                   Load_Symbol=Load_Symbol, he_init=he_init, biases=biases)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs,
                  ST_Load_filters, ST_Load_biases ,
                  N1_load_offset,N1_load_scale,N1_load_moving_mean,N1_load_moving_variance,
                  Con1_Load_filters, Con1_Load_biases,
                  N2_load_offset,N2_load_scale,N2_load_moving_mean,N2_load_moving_variance,
                  Con2_Load_filters, Con2_Load_biases,
                  Load_Symbol,
                  resample=None, no_dropout=False, labels=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample=='up':
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', Load_filters=ST_Load_filters, Load_biases=ST_Load_biases,
                                 Load_Symbol=Load_Symbol, input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.N1', output, labels=labels, load_offset=N1_load_offset, load_scale=N1_load_scale,
                       load_moving_mean=N1_load_moving_mean, load_moving_variance=N1_load_moving_variance, load_mark=Load_Symbol)

    output = nonlinearity(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output,
                    Load_filters=Con1_Load_filters, Load_biases=Con1_Load_biases , Load_Symbol=Load_Symbol)

    output = Normalize(name+'.N2', output, labels=labels, load_offset=N2_load_offset, load_scale=N2_load_scale,
                       load_moving_mean=N2_load_moving_mean, load_moving_variance=N2_load_moving_variance, load_mark=Load_Symbol)

    output = nonlinearity(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output,
                    Load_filters=Con2_Load_filters, Load_biases=Con2_Load_biases , Load_Symbol=Load_Symbol)

    return shortcut + output

def OptimizedResBlockDisc1(inputs,
                  ST_Load_filters, ST_Load_biases ,
                  Con1_Load_filters, Con1_Load_biases,
                  Con2_Load_filters, Con2_Load_biases,
                  Load_Symbol,):
    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=3, output_dim=DIM_D)
    conv_2        = functools.partial(ConvMeanPool, input_dim=DIM_D, output_dim=DIM_D)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut('Discriminator.1.Shortcut', input_dim=3, output_dim=DIM_D, filter_size=1,
                             Load_filters=ST_Load_filters, Load_biases=ST_Load_biases, Load_Symbol=Load_Symbol,
                             he_init=False,biases=True, inputs=inputs)

    output = inputs
    output = conv_1('Discriminator.1.Conv1', filter_size=3, inputs=output,
                    Load_filters=Con1_Load_filters, Load_biases=Con1_Load_biases, Load_Symbol=Load_Symbol)
    output = nonlinearity(output)
    output = conv_2('Discriminator.1.Conv2', filter_size=3, inputs=output,
                    Load_filters=Con2_Load_filters, Load_biases=Con2_Load_biases, Load_Symbol=Load_Symbol)
    return shortcut + output
#G载入变量占位符，共2+10+10+10+4+2=38个
G_Input_W = tf.placeholder(tf.float32, shape=[128, 4 * 4 * DIM_G])
G_Input_B = tf.placeholder(tf.float32, shape=[4 * 4 * DIM_G, ])

G_L1_ST_F = tf.placeholder(tf.float32, shape=[1, 1, DIM_G, DIM_G])
G_L1_ST_B = tf.placeholder(tf.float32, shape=[DIM_G, ])

G_L1_CON1_F = tf.placeholder(tf.float32, shape=[3, 3, DIM_G, DIM_G])
G_L1_CON1_B = tf.placeholder(tf.float32, shape=[DIM_G, ])
G_L1_CON2_F = tf.placeholder(tf.float32, shape=[3, 3, DIM_G, DIM_G])
G_L1_CON2_B = tf.placeholder(tf.float32, shape=[DIM_G, ])

G_L2_ST_F = tf.placeholder(tf.float32, shape=[1, 1, DIM_G, DIM_G])
G_L2_ST_B = tf.placeholder(tf.float32, shape=[DIM_G, ])
G_L2_CON1_F = tf.placeholder(tf.float32, shape=[3, 3, DIM_G, DIM_G])
G_L2_CON1_B = tf.placeholder(tf.float32, shape=[DIM_G, ])
G_L2_CON2_F = tf.placeholder(tf.float32, shape=[3, 3, DIM_G, DIM_G])
G_L2_CON2_B = tf.placeholder(tf.float32, shape=[DIM_G, ])

G_L3_ST_F = tf.placeholder(tf.float32, shape=[1, 1, DIM_G, DIM_G])
G_L3_ST_B = tf.placeholder(tf.float32, shape=[DIM_G, ])
G_L3_CON1_F = tf.placeholder(tf.float32, shape=[3, 3, DIM_G, DIM_G])
G_L3_CON1_B = tf.placeholder(tf.float32, shape=[DIM_G, ])
G_L3_CON2_F = tf.placeholder(tf.float32, shape=[3, 3, DIM_G, DIM_G])
G_L3_CON2_B = tf.placeholder(tf.float32, shape=[DIM_G, ])

G_OUTN_O = tf.placeholder(tf.float32, shape=[DIM_G, ])
G_OUTN_S = tf.placeholder(tf.float32, shape=[DIM_G, ])
G_OUTN_MM = tf.placeholder(tf.float32, shape=[DIM_G, ])
G_OUTN_MV = tf.placeholder(tf.float32, shape=[DIM_G, ])

G_OUTPUT_F = tf.placeholder(tf.float32, shape=[3, 3, DIM_G, 3])
G_OUTPUT_B = tf.placeholder(tf.float32, shape=[3, ])
if CONDITIONAL:
    G_L1_N1_O = tf.placeholder(tf.float32, shape=[10, DIM_G])
    G_L1_N1_S = tf.placeholder(tf.float32, shape=[10, DIM_G])
    G_L1_N1_MM = np.zeros([10, DIM_G])
    G_L1_N1_MV = np.zeros([10, DIM_G])
    G_L1_N2_O = tf.placeholder(tf.float32, shape=[10, DIM_G])
    G_L1_N2_S = tf.placeholder(tf.float32, shape=[10, DIM_G])
    G_L1_N2_MM = np.zeros([10, DIM_G])
    G_L1_N2_MV = np.zeros([10, DIM_G])

    G_L2_N1_O = tf.placeholder(tf.float32, shape=[10, DIM_G])
    G_L2_N1_S = tf.placeholder(tf.float32, shape=[10, DIM_G])
    G_L2_N1_MM = np.zeros([10, DIM_G])
    G_L2_N1_MV = np.zeros([10, DIM_G])
    G_L2_N2_O = tf.placeholder(tf.float32, shape=[10, DIM_G])
    G_L2_N2_S = tf.placeholder(tf.float32, shape=[10, DIM_G])
    G_L2_N2_MM = np.zeros([10, DIM_G])
    G_L2_N2_MV = np.zeros([10, DIM_G])

    G_L3_N1_O = tf.placeholder(tf.float32, shape=[10, DIM_G])
    G_L3_N1_S = tf.placeholder(tf.float32, shape=[10, DIM_G])
    G_L3_N1_MM = np.zeros([10, DIM_G])
    G_L3_N1_MV = np.zeros([10, DIM_G])
    G_L3_N2_O = tf.placeholder(tf.float32, shape=[10, DIM_G])
    G_L3_N2_S = tf.placeholder(tf.float32, shape=[10, DIM_G])
    G_L3_N2_MM = np.zeros([10, DIM_G])
    G_L3_N2_MV = np.zeros([10, DIM_G])

    D_ACG_OUT_W = tf.placeholder(tf.float32, shape=[DIM_D, 10])
    D_ACG_OUT_B = tf.placeholder(tf.float32, shape=[10, ])
else:
    G_L1_N1_O = tf.placeholder(tf.float32, shape=[DIM_G, ])
    G_L1_N1_S = tf.placeholder(tf.float32, shape=[DIM_G, ])
    G_L1_N1_MM = tf.placeholder(tf.float32, shape=[DIM_G, ])
    G_L1_N1_MV = tf.placeholder(tf.float32, shape=[DIM_G, ])

    G_L1_N2_O = tf.placeholder(tf.float32, shape=[DIM_G, ])
    G_L1_N2_S = tf.placeholder(tf.float32, shape=[DIM_G, ])
    G_L1_N2_MM = tf.placeholder(tf.float32, shape=[DIM_G, ])
    G_L1_N2_MV = tf.placeholder(tf.float32, shape=[DIM_G, ])

    G_L2_N1_O = tf.placeholder(tf.float32, shape=[DIM_G, ])
    G_L2_N1_S = tf.placeholder(tf.float32, shape=[DIM_G, ])
    G_L2_N1_MM = tf.placeholder(tf.float32, shape=[DIM_G, ])
    G_L2_N1_MV = tf.placeholder(tf.float32, shape=[DIM_G, ])

    G_L2_N2_O = tf.placeholder(tf.float32, shape=[DIM_G, ])
    G_L2_N2_S = tf.placeholder(tf.float32, shape=[DIM_G, ])
    G_L2_N2_MM = tf.placeholder(tf.float32, shape=[DIM_G, ])
    G_L2_N2_MV = tf.placeholder(tf.float32, shape=[DIM_G, ])

    G_L3_N1_O = tf.placeholder(tf.float32, shape=[DIM_G, ])
    G_L3_N1_S = tf.placeholder(tf.float32, shape=[DIM_G, ])
    G_L3_N1_MM = tf.placeholder(tf.float32, shape=[DIM_G, ])
    G_L3_N1_MV = tf.placeholder(tf.float32, shape=[DIM_G, ])

    G_L3_N2_O = tf.placeholder(tf.float32, shape=[DIM_G, ])
    G_L3_N2_S = tf.placeholder(tf.float32, shape=[DIM_G, ])
    G_L3_N2_MM = tf.placeholder(tf.float32, shape=[DIM_G, ])
    G_L3_N2_MV = tf.placeholder(tf.float32, shape=[DIM_G, ])


def Generator(n_samples, labels, noise=None):
    if noise is None:
        noise = tf.random_normal([int(n_samples), 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*DIM_G, noise, G_Input_W, G_Input_B, Load_Symbol=G_SYMBOL)
    output = tf.reshape(output, [-1, DIM_G, 4, 4])
    output = ResidualBlock('Generator.1', DIM_G, DIM_G, 3, output,
                           G_L1_ST_F, G_L1_ST_B, G_L1_N1_O, G_L1_N1_S, G_L1_N1_MM, G_L1_N1_MV,
                           G_L1_CON1_F, G_L1_CON1_B, G_L1_N2_O, G_L1_N2_S, G_L1_N2_MM, G_L1_N2_MV,
                           G_L1_CON2_F, G_L1_CON2_B, Load_Symbol=G_SYMBOL,
                           resample='up', labels=labels)
    output = ResidualBlock('Generator.2', DIM_G, DIM_G, 3, output,
                           G_L2_ST_F, G_L2_ST_B, G_L2_N1_O, G_L2_N1_S, G_L2_N1_MM, G_L2_N1_MV,
                           G_L2_CON1_F, G_L2_CON1_B, G_L2_N2_O, G_L2_N2_S, G_L2_N2_MM, G_L2_N2_MV,
                           G_L2_CON2_F, G_L2_CON2_B, Load_Symbol=G_SYMBOL,
                           resample='up', labels=labels)
    output = ResidualBlock('Generator.3', DIM_G, DIM_G, 3, output,
                           G_L3_ST_F, G_L3_ST_B, G_L3_N1_O, G_L3_N1_S, G_L3_N1_MM, G_L3_N1_MV,
                           G_L3_CON1_F, G_L3_CON1_B, G_L3_N2_O, G_L3_N2_S, G_L3_N2_MM, G_L3_N2_MV,
                           G_L3_CON2_F, G_L3_CON2_B, Load_Symbol=G_SYMBOL, resample='up', labels=labels)
    output = Normalize('Generator.OutputN', output, G_OUTN_O, G_OUTN_S, G_OUTN_MM, G_OUTN_MV, load_mark=G_SYMBOL)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G, 3, 3, output, G_OUTPUT_F, G_OUTPUT_B, Load_Symbol=G_SYMBOL, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, OUTPUT_DIM])
#D载入变量占位符，共6+6+4+4+2+2=24个
D_L1_ST_F = tf.placeholder(tf.float32, shape=[1, 1, 3, DIM_D])
D_L1_ST_B = tf.placeholder(tf.float32, shape=[DIM_D, ])
D_L1_CON1_F = tf.placeholder(tf.float32, shape=[3, 3, 3, DIM_D])
D_L1_CON1_B = tf.placeholder(tf.float32, shape=[DIM_D, ])
D_L1_CON2_F = tf.placeholder(tf.float32, shape=[3, 3, 3, DIM_D])
D_L1_CON2_B = tf.placeholder(tf.float32, shape=[DIM_D, ])

D_L2_ST_F = tf.placeholder(tf.float32, shape=[1, 1, DIM_D, DIM_D])
D_L2_ST_B = tf.placeholder(tf.float32, shape=[DIM_D, ])
D_L2_CON1_F = tf.placeholder(tf.float32, shape=[3, 3, DIM_D, DIM_D])
D_L2_CON1_B = tf.placeholder(tf.float32, shape=[DIM_D, ])
D_L2_CON2_F = tf.placeholder(tf.float32, shape=[3, 3, DIM_D, DIM_D])
D_L2_CON2_B = tf.placeholder(tf.float32, shape=[DIM_D, ])

D_L3_CON1_F = tf.placeholder(tf.float32, shape=[3, 3, DIM_D, DIM_D])
D_L3_CON1_B = tf.placeholder(tf.float32, shape=[DIM_D, ])
D_L3_CON2_F = tf.placeholder(tf.float32, shape=[3, 3, DIM_D, DIM_D])
D_L3_CON2_B = tf.placeholder(tf.float32, shape=[DIM_D, ])

D_L4_CON1_F = tf.placeholder(tf.float32, shape=[3, 3, DIM_D, DIM_D])
D_L4_CON1_B = tf.placeholder(tf.float32, shape=[DIM_D, ])
D_L4_CON2_F = tf.placeholder(tf.float32, shape=[3, 3, DIM_D, DIM_D])
D_L4_CON2_B = tf.placeholder(tf.float32, shape=[DIM_D, ])

D_OUTPUT_W = tf.placeholder(tf.float32, shape=[DIM_D, 1])
D_OUTPUT_B = tf.placeholder(tf.float32, shape=[1, ])


def Discriminator(inputs, labels):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = OptimizedResBlockDisc1(output, D_L1_ST_F, D_L1_ST_B, D_L1_CON1_F, D_L1_CON1_B, D_L1_CON2_F, D_L1_CON2_B,
                                    Load_Symbol=D_SYMBOL)

    output = ResidualBlock('Discriminator.2', DIM_D, DIM_D, 3, output, D_L2_ST_F, D_L2_ST_B,
                           np.zeros([10, DIM_D]), np.zeros([10, DIM_D]), np.zeros([10, DIM_D]), np.zeros([10, DIM_D]),
                           D_L2_CON1_F, D_L2_CON1_B,
                           np.zeros([10, DIM_D]), np.zeros([10, DIM_D]), np.zeros([10, DIM_D]), np.zeros([10, DIM_D]),
                           D_L2_CON2_F, D_L2_CON2_B, Load_Symbol=D_SYMBOL,
                           resample='down', labels=labels)

    output = ResidualBlock('Discriminator.3', DIM_D, DIM_D, 3, output, np.zeros([1, 1, DIM_D, DIM_D]), np.zeros([1, 1, DIM_D, DIM_D]),
                           np.zeros([10, DIM_D]), np.zeros([10, DIM_D]), np.zeros([10, DIM_D]), np.zeros([10, DIM_D]),
                           D_L3_CON1_F, D_L3_CON1_B,
                           np.zeros([10, DIM_D]), np.zeros([10, DIM_D]), np.zeros([10, DIM_D]), np.zeros([10, DIM_D]),
                           D_L3_CON2_F, D_L3_CON2_B, Load_Symbol=D_SYMBOL,
                           resample=None, labels=labels)

    output = ResidualBlock('Discriminator.4', DIM_D, DIM_D, 3, output, np.zeros([1, 1, DIM_D, DIM_D]), np.zeros([1, 1, DIM_D, DIM_D]),
                           np.zeros([10, DIM_D]), np.zeros([10, DIM_D]), np.zeros([10, DIM_D]), np.zeros([10, DIM_D]),
                           D_L4_CON1_F, D_L4_CON1_B,
                           np.zeros([10, DIM_D]), np.zeros([10, DIM_D]), np.zeros([10, DIM_D]), np.zeros([10, DIM_D]),
                           D_L4_CON2_F, D_L4_CON2_B, Load_Symbol=D_SYMBOL,
                           resample=None, labels=labels)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2,3])
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', DIM_D, 1, output, D_OUTPUT_W, D_OUTPUT_B, Load_Symbol=D_SYMBOL)
    output_wgan = tf.reshape(output_wgan, [-1])
    if CONDITIONAL and ACGAN:
        output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', DIM_D, 10, output, D_ACG_OUT_W, D_ACG_OUT_B, Load_Symbol=D_SYMBOL)
        return output_wgan, output_acgan
    else:
        return output_wgan, None
ckpt_gen_path = './resnet_gen_cifar_path'

ckpt_disc_path = './resnet_disc_cifar_path'


'''def tensorflow_load():
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
      '''


    #tf.reset_default_graph()
    #G_SYMBOL= G_SYMBOL_test
    #D_SYMBOL= D_SYMBOL_test
with tf.Session() as session:
        _iteration: object = tf.placeholder(tf.int32, shape=None)
        all_real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
        all_real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

        labels_splits = tf.split(all_real_labels, len(DEVICES), axis=0)

        fake_data_splits = []
        for i, device in enumerate(DEVICES):
            with tf.device(device):
                fake_data_splits.append(Generator(int(BATCH_SIZE / len(DEVICES)), labels_splits[i]))

        all_real_data = tf.reshape(2 * ((tf.cast(all_real_data_int, tf.float32) / 256.) - .5), [BATCH_SIZE, OUTPUT_DIM])
        all_real_data += tf.random_uniform(shape=[BATCH_SIZE, OUTPUT_DIM], minval=0., maxval=1. / 128)  # dequantize
        all_real_data_splits = tf.split(all_real_data, len(DEVICES), axis=0)

        print(np.shape(DEVICES), DEVICES)

        DEVICES_B = DEVICES[:int(len(DEVICES) / 2)]
        DEVICES_A = DEVICES[int(len(DEVICES) / 2):]

        disc_costs = []
        disc_acgan_costs = []
        disc_acgan_accs = []
        disc_acgan_fake_accs = []
        for i, device in enumerate(DEVICES_A):
            with tf.device(device):
                real_and_fake_data = tf.concat([
                    all_real_data_splits[i],
                    all_real_data_splits[len(DEVICES_A) + i],
                    fake_data_splits[i],
                    fake_data_splits[len(DEVICES_A) + i]
                ], axis=0)
                real_and_fake_labels = tf.concat([
                    labels_splits[i],
                    labels_splits[len(DEVICES_A) + i],
                    labels_splits[i],
                    labels_splits[len(DEVICES_A) + i]
                ], axis=0)
                disc_all, disc_all_acgan = Discriminator(real_and_fake_data, real_and_fake_labels)
                disc_real = disc_all[:int(BATCH_SIZE / len(DEVICES_A))]
                disc_fake = disc_all[int(BATCH_SIZE / len(DEVICES_A)):]
                disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))
                if CONDITIONAL and ACGAN:
                    disc_acgan_costs.append(tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=disc_all_acgan[:int(BATCH_SIZE / len(DEVICES_A))],
                            labels=real_and_fake_labels[:int(BATCH_SIZE / len(DEVICES_A))])
                    ))
                    disc_acgan_accs.append(tf.reduce_mean(
                        tf.cast(
                            tf.equal(
                                tf.to_int32(tf.argmax(disc_all_acgan[:int(BATCH_SIZE / len(DEVICES_A))], dimension=1)),
                                real_and_fake_labels[:int(BATCH_SIZE / len(DEVICES_A))]
                            ),
                            tf.float32
                        )
                    ))
                    disc_acgan_fake_accs.append(tf.reduce_mean(
                        tf.cast(
                            tf.equal(
                                tf.to_int32(tf.argmax(disc_all_acgan[int(BATCH_SIZE / len(DEVICES_A)):], dimension=1)),
                                real_and_fake_labels[int(BATCH_SIZE / len(DEVICES_A)):]
                            ),
                            tf.float32
                        )
                    ))

        for i, device in enumerate(DEVICES_B):
            with tf.device(device):
                real_data = tf.concat([all_real_data_splits[i], all_real_data_splits[len(DEVICES_A) + i]], axis=0)
                fake_data = tf.concat([fake_data_splits[i], fake_data_splits[len(DEVICES_A) + i]], axis=0)
                labels = tf.concat([
                    labels_splits[i],
                    labels_splits[len(DEVICES_A) + i],
                ], axis=0)
                alpha = tf.random_uniform(
                    shape=[int(BATCH_SIZE / len(DEVICES_A)), 1],
                    minval=0.,
                    maxval=1.
                )
                differences = fake_data - real_data
                interpolates = real_data + (alpha * differences)
                gradients = tf.gradients(Discriminator(interpolates, labels)[0], [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                gradient_penalty = 10 * tf.reduce_mean((slopes - 1.) ** 2)
                disc_costs.append(gradient_penalty)

        disc_wgan = tf.add_n(disc_costs) / len(DEVICES_A)
        if CONDITIONAL and ACGAN:
            disc_acgan = tf.add_n(disc_acgan_costs) / len(DEVICES_A)
            disc_acgan_acc = tf.add_n(disc_acgan_accs) / len(DEVICES_A)
            disc_acgan_fake_acc = tf.add_n(disc_acgan_fake_accs) / len(DEVICES_A)
            disc_cost = disc_wgan + (ACGAN_SCALE * disc_acgan)
        else:
            disc_acgan = tf.constant(0.)
            disc_acgan_acc = tf.constant(0.)
            disc_acgan_fake_acc = tf.constant(0.)
            disc_cost = disc_wgan

        disc_params = lib.params_with_name('Discriminator.')
        gen_params = lib.params_with_name('Generator')

        print("G参数", np.shape(gen_params), gen_params)
        print("D参数", np.shape(disc_params), disc_params)
        # 保存模型信息


        global_step = tf.Variable(0, name='global_step', trainable=False)

        if DECAY:
            decay = tf.maximum(0.2, 1. - (tf.cast(_iteration, tf.float32) / ITERS))
        else:
            decay = 1.

        gen_costs = []
        gen_acgan_costs = []
        for device in DEVICES:
            with tf.device(device):
                n_samples = GEN_BS_MULTIPLE * BATCH_SIZE / len(DEVICES)
                fake_labels = tf.cast(tf.random_uniform([int(n_samples)]) * 10, tf.int32)
                if CONDITIONAL and ACGAN:
                    disc_fake, disc_fake_acgan = Discriminator(Generator(int(n_samples), fake_labels), fake_labels)
                    gen_costs.append(-tf.reduce_mean(disc_fake))
                    gen_acgan_costs.append(tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels)
                    ))
                else:
                    gen_costs.append(-tf.reduce_mean(Discriminator(Generator(n_samples, fake_labels), fake_labels)[0]))
        gen_cost = (tf.add_n(gen_costs) / len(DEVICES))
        if CONDITIONAL and ACGAN:
            gen_cost += (ACGAN_SCALE_G * (tf.add_n(gen_acgan_costs) / len(DEVICES)))

        gen_opt = tf.train.AdamOptimizer(learning_rate=LR * decay, beta1=0., beta2=0.9)
        disc_opt = tf.train.AdamOptimizer(learning_rate=LR * decay, beta1=0., beta2=0.9)
        gen_gv = gen_opt.compute_gradients(gen_cost, var_list=gen_params)
        disc_gv = disc_opt.compute_gradients(disc_cost, var_list=disc_params)
        gen_train_op = gen_opt.apply_gradients(gen_gv)
        disc_train_op = disc_opt.apply_gradients(disc_gv)

        # Function for generating samples
        frame_i = [0]
        fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
        fixed_labels = tf.constant(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10, dtype='int32'))
        fixed_noise_samples = Generator(100, fixed_labels, noise=fixed_noise)
        # Function for calculating inception score
        fake_labels_100 = tf.cast(tf.random_uniform([100]) * 10, tf.int32)
        samples_100 = Generator(100, fake_labels_100)

        def get_inception_score(n):
            all_samples = []
            for i in range(int(n / 100)):
                all_samples.append(session.run(samples_100))
                all_samples = np.concatenate(all_samples, axis=0)
                all_samples = ((all_samples + 1.) * (255.99 / 2)).astype('int32')
                all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
                return lib.inception_score.get_inception_score(list(all_samples))

        for name, grads_and_vars in [('G', gen_gv), ('D', disc_gv)]:
            print("{} Params:".format(name))
            total_param_count = 0
            for g, v in grads_and_vars:
                shape = v.get_shape()
                shape_str = ",".join([str(x) for x in v.get_shape()])

                param_count = 1
                for dim in shape:
                    param_count *= int(dim)
                total_param_count += param_count

                if g == None:
                    print("\t{} ({}) [no grad!]".format(v.name, shape_str))
                else:
                    print("\t{} ({})".format(v.name, shape_str))
            print("Total param count: {}".format(
                locale.format("%d", total_param_count, grouping=True)
            ))
        resnet_image_save_path = './resnet_cifar_image_save_path'
        if not os.path.exists(resnet_image_save_path):
            os.makedirs(resnet_image_save_path)

        session.run(tf.initialize_all_variables())

        for anti_label in range(10 - start_point):

            if LOAD_MARK:
                print("读取生成器模型及参数")
                # 生成器：
                G_SYMBOL = True


                '''gen_saver = tf.train.import_meta_graph(
                    ckpt_gen_path + '/resnet_model_gen_cifar_ckpt-{}.meta'.format(model_number))
                gen_saver.restore(session, tf.train.latest_checkpoint(ckpt_gen_path))
                '''
                G_Input_W = session.run('Generator.Input/Generator.Input.W:0')
                G_Input_B = session.run('Generator.Input/Generator.Input.b:0')

                G_L1_ST_F = session.run('Generator.1.Shortcut/Generator.1.Shortcut.Filters:0')
                G_L1_ST_B = session.run('Generator.1.Shortcut/Generator.1.Shortcut.Biases:0')
                G_L1_N1_O = session.run('Generator.1.N1.offset:0')
                G_L1_N1_S = session.run('Generator.1.N1.scale:0')
                G_L1_CON1_F = session.run('Generator.1.Conv1/Generator.1.Conv1.Filters:0')
                G_L1_CON1_B = session.run('Generator.1.Conv1/Generator.1.Conv1.Biases:0')
                G_L1_N2_O = session.run('Generator.1.N2.offset:0')
                G_L1_N2_S = session.run('Generator.1.N2.scale:0')
                G_L1_CON2_F = session.run('Generator.1.Conv2/Generator.1.Conv2.Filters:0')
                G_L1_CON2_B = session.run('Generator.1.Conv2/Generator.1.Conv2.Biases:0')

                G_L2_ST_F = session.run('Generator.2.Shortcut/Generator.2.Shortcut.Filters:0')
                G_L2_ST_B = session.run('Generator.2.Shortcut/Generator.2.Shortcut.Biases:0')
                G_L2_N1_O = session.run('Generator.2.N1.offset:0')
                G_L2_N1_S = session.run('Generator.2.N1.scale:0')
                G_L2_CON1_F = session.run('Generator.2.Conv1/Generator.2.Conv1.Filters:0')
                G_L2_CON1_B = session.run('Generator.2.Conv1/Generator.2.Conv1.Biases:0')
                G_L2_N2_O = session.run('Generator.2.N2.offset:0')
                G_L2_N2_S = session.run('Generator.2.N2.scale:0')
                G_L2_CON2_F = session.run('Generator.2.Conv2/Generator.2.Conv2.Filters:0')
                G_L2_CON2_B = session.run('Generator.2.Conv2/Generator.2.Conv2.Biases:0')

                G_L3_ST_F = session.run('Generator.3.Shortcut/Generator.3.Shortcut.Filters:0')
                G_L3_ST_B = session.run('Generator.3.Shortcut/Generator.3.Shortcut.Biases:0')
                G_L3_N1_O = session.run('Generator.3.N1.offset:0')
                G_L3_N1_S = session.run('Generator.3.N1.scale:0')
                G_L3_CON1_F = session.run('Generator.3.Conv1/Generator.3.Conv1.Filters:0')
                G_L3_CON1_B = session.run('Generator.3.Conv1/Generator.3.Conv1.Biases:0')
                G_L3_N2_O = session.run('Generator.3.N2.offset:0')
                G_L3_N2_S = session.run('Generator.3.N2.scale:0')
                G_L3_CON2_F = session.run('Generator.3.Conv2/Generator.3.Conv2.Filters:0')
                G_L3_CON2_B = session.run('Generator.3.Conv2/Generator.3.Conv2.Biases:0')

                G_OUTN_O = session.run('Generator.OutputN.offset:0')
                G_OUTN_S = session.run('Generator.OutputN.scale:0')
                G_OUTN_MM = session.run('Generator.OutputN.moving_mean:0')
                G_OUTN_MV = session.run('Generator.OutputN.moving_variance:0')

                G_OUTPUT_F = session.run('Generator.Output/Generator.Output.Filters:0')
                G_OUTPUT_B = session.run('Generator.Output/Generator.Output.Biases:0')
                del (gen_saver)
                # tf.reset_default_graph()
                # 判别器：
                print("读取判别器模型及参数")
                D_SYMBOL = True
                dis_saver = tf.train.import_meta_graph(
                    ckpt_disc_path + '/resnet_model_disc_cifar_ckpt-{}.meta'.format(model_number))
                dis_saver.restore(session, tf.train.latest_checkpoint(ckpt_disc_path))

                D_L1_ST_F = session.run('Discriminator.1.Shortcut/Discriminator.1.Shortcut.Filters:0')
                D_L1_ST_B = session.run('Discriminator.1.Shortcut/Discriminator.1.Shortcut.Biases:0')
                D_L1_CON1_F = session.run('Discriminator.1.Conv1/Discriminator.1.Conv1.Filters:0')
                D_L1_CON1_B = session.run('Discriminator.1.Conv1/Discriminator.1.Conv1.Biases:0')
                D_L1_CON2_F = session.run('Discriminator.1.Conv2/Discriminator.1.Conv2.Filters:0')
                D_L1_CON2_B = session.run('Discriminator.1.Conv2/Discriminator.1.Conv2.Biases:0')

                D_L2_ST_F = session.run('Discriminator.2.Shortcut/Discriminator.2.Shortcut.Filters:0')
                D_L2_ST_B = session.run('Discriminator.2.Shortcut/Discriminator.2.Shortcut.Biases:0')
                D_L2_CON1_F = session.run('Discriminator.2.Conv1/Discriminator.2.Conv1.Filters:0')
                D_L2_CON1_B = session.run('Discriminator.2.Conv1/Discriminator.2.Conv1.Biases:0')
                D_L2_CON2_F = session.run('Discriminator.2.Conv2/Discriminator.2.Conv2.Filters:0')
                D_L2_CON2_B = session.run('Discriminator.2.Conv2/Discriminator.2.Conv2.Biases:0')

                D_L3_CON1_F = session.run('Discriminator.3.Conv1/Discriminator.3.Conv1.Filters:0')
                D_L3_CON1_B = session.run('Discriminator.3.Conv1/Discriminator.3.Conv1.Biases:0')
                D_L3_CON2_F = session.run('Discriminator.3.Conv2/Discriminator.3.Conv2.Filters:0')
                D_L3_CON2_B = session.run('Discriminator.3.Conv2/Discriminator.3.Conv2.Biases:0')

                D_L4_CON1_F = session.run('Discriminator.4.Conv1/Discriminator.4.Conv1.Filters:0')
                D_L4_CON1_B = session.run('Discriminator.4.Conv1/Discriminator.4.Conv1.Biases:0')
                D_L4_CON2_F = session.run('Discriminator.4.Conv2/Discriminator.4.Conv2.Filters:0')
                D_L4_CON2_B = session.run('Discriminator.4.Conv2/Discriminator.4.Conv2.Biases:0')

                D_OUTPUT_W = session.run('Discriminator.Output/Discriminator.Output.W:0')
                D_OUTPUT_B = session.run('Discriminator.Output/Discriminator.Output.b:0')
                del (dis_saver)
                # tf.reset_default_graph()
                if CONDITIONAL:
                    D_ACG_OUT_W = session.run('Discriminator.ACGANOutput/Discriminator.ACGANOutput.W:0')
                    D_ACG_OUT_B = session.run('Discriminator.ACGANOutput/Discriminator.ACGANOutput.b:0')
                else:
                    G_L1_N1_MM = session.run('Generator.1.N1.moving_mean:0')
                    G_L1_N1_MV = session.run('Generator.1.N1.moving_variance:0')
                    G_L1_N2_MM = session.run('Generator.1.N2.moving_mean:0')
                    G_L1_N2_MV = session.run('Generator.1.N2.moving_variance:0')

                    G_L2_N1_MM = session.run('Generator.2.N1.moving_mean:0')
                    G_L2_N1_MV = session.run('Generator.2.N1.moving_variance:0')
                    G_L2_N2_MM = session.run('Generator.2.N2.moving_mean:0')
                    G_L2_N2_MV = session.run('Generator.2.N2.moving_variance:0')

                    G_L3_N1_MM = session.run('Generator.3.N1.moving_mean:0')
                    G_L3_N1_MV = session.run('Generator.3.N1.moving_variance:0')
                    G_L3_N2_MM = session.run('Generator.3.N2.moving_mean:0')
                    G_L3_N2_MV = session.run('Generator.3.N2.moving_variance:0')

            anti_label = anti_label + start_point

            cifar_image_save_path = resnet_image_save_path + './ant1_label_{}'.format(anti_label)
            if not os.path.exists(cifar_image_save_path):
                os.makedirs(cifar_image_save_path)
            ckpt_gen_path = './resnet_gen_cifar_path'
            ckpt_gen_cifar_path_new = './resnet_gen_cifar_path_new/anti_label_{}'.format(anti_label)
            if not os.path.exists(ckpt_gen_cifar_path_new):
                os.makedirs(ckpt_gen_cifar_path_new)

            ckpt_disc_path = './resnet_disc_cifar_path'
            ckpt_disc_cifar_path_new = './resnet_disc_cifar_path_new/anti_label_{}'.format(anti_label)
            if not os.path.exists(ckpt_disc_cifar_path_new):
                os.makedirs(ckpt_disc_cifar_path_new)

            def generate_image(frame, true_dist, image_path):
                samples = session.run(fixed_noise_samples)
                samples = ((samples + 1.) * (255. / 2)).astype('int32')
                lib.save_images.save_images(samples.reshape((100, 3, 32, 32)),
                                            image_path + '/samples_{}.png'.format(frame))
                return samples

            saver_gen = tf.train.Saver(gen_params, max_to_keep=1)
            saver_disc = tf.train.Saver(disc_params, max_to_keep=1)

            train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, DATA_DIR, anti_label, minus_value)

            # test_images, test_labels = train_gen
            # print("shape of test_images:",np.shape(test_images),"\n shape of test_labels",np.shape(test_labels))
            # print(np.sum(test_labels,axis=0))

            def inf_train_gen():
                while True:
                    for images, _labels in train_gen():
                        yield images, _labels

            if anti_label != 1:
                gen = inf_train_gen()

                print("anti_label is:", anti_label)
                cpz = 0
                for iteration in range(ST_ITERS):
                    start_time = time.time()

                    #if LOAD_MARK:  # 加载完一次后就将MARK置为False
                        #LOAD_MARK = False

                    if iteration > 0:
                        _ = session.run([gen_train_op], feed_dict={_iteration: iteration})

                    if G_SYMBOL:  # 生成器训练一次后就将G_SYMBOL置为False
                        G_SYMBOL = False
                        '''del(G_Input_W, G_L1_ST_F, G_L1_CON1_F, G_L1_CON2_F,
                            G_L2_ST_F, G_L2_CON1_F, G_L2_CON2_F,
                            G_L3_ST_F, G_L3_CON1_F, G_L3_CON2_F, G_OUTPUT_F)

                        '''
                        '''G_L1_ST_F.clear()
                        G_L1_CON1_F.clear()
                        G_L1_CON2_F.clear()
                        G_L2_ST_F.clear()
                        G_L2_CON1_F.clear()
                        G_L2_CON2_F.clear()
                        G_L3_ST_F.clear()
                        G_L3_CON1_F.clear()
                        G_L3_CON2_F.clear()
                        G_OUTPUT_F.clear()'''

                    for i in range(N_CRITIC):
                        _data, _labels = gen.__next__()
                        '''
                        print("shape of test_images:",np.shape(_data),"\n shape of test_labels",np.shape(_labels))
                        print(np.sum(_labels,axis=0))
                        test_images_path = cifar_image_save_path + './test{}'.format(anti_label)
                        if not os.path.exists(test_images_path):
                            os.makedirs(test_images_path)

                        lib.save_images.save_images(_data.reshape((64, 3, 32, 32)),
                                                    test_images_path + '/samples_{}.png'.format(cpz))
                        cpz += 1.
                        '''
                        label_test_mark = np.sum(_labels, axis=0)
                        label_test_mark = 0.5 - label_test_mark / (2 * BATCH_SIZE)
                        # print(label_test_mark)
                        while (label_test_mark < minus_value / 3):
                            _data, _labels = gen.__next__()
                            label_test_mark = np.sum(_labels, axis=0)
                            label_test_mark = 0.5 - label_test_mark / (2 * BATCH_SIZE)
                            # print(label_test_mark)
                        # print("bbbbbb")

                        if CONDITIONAL and ACGAN:
                            _disc_cost, _disc_wgan, _disc_acgan, _disc_acgan_acc, _disc_acgan_fake_acc, _ = session.run(
                                [disc_cost,
                                 disc_wgan, disc_acgan, disc_acgan_acc, disc_acgan_fake_acc, disc_train_op],
                                feed_dict={all_real_data_int: _data, all_real_labels: _labels, _iteration: iteration})
                        else:
                            _disc_cost, _ = session.run([disc_cost, disc_train_op],
                                                        feed_dict={all_real_data_int: _data, all_real_labels: _labels,
                                                                   _iteration: iteration})
                        if D_SYMBOL:  # 判别器训练一次后就将D_SYMBOL置为False
                            D_SYMBOL = False
                            '''del(D_L1_ST_F, D_L1_CON1_F, D_L1_CON2_F,
                                D_L2_ST_F, D_L2_CON1_F, D_L2_CON2_F,
                                D_L3_CON1_F, D_L3_CON2_F,D_L4_CON1_F, D_L4_CON2_F,D_OUTPUT_W)
                            D_L1_ST_F.clear()
                            D_L1_CON1_F.clear()
                            D_L1_CON2_F.clear()
                            D_L2_ST_F.clear()
                            D_L2_CON1_F.clear()
                            D_L2_CON2_F.clear()
                            D_L3_CON1_F.clear()
                            D_L3_CON2_F.clear()
                            D_L4_CON1_F.clear()
                            D_L4_CON2_F.clear()
                            D_OUTPUT_W.clear()'''

                    lib.plot.plot('cost', _disc_cost)
                    if CONDITIONAL and ACGAN:
                        lib.plot.plot('wgan', _disc_wgan)
                        lib.plot.plot('acgan', _disc_acgan)
                        lib.plot.plot('acc_real', _disc_acgan_acc)
                        lib.plot.plot('acc_fake', _disc_acgan_fake_acc)
                    lib.plot.plot('time', time.time() - start_time)

                    if iteration % INCEPTION_FREQUENCY == INCEPTION_FREQUENCY - 1:
                        inception_score = get_inception_score(50000)
                        lib.plot.plot('inception_50k', inception_score[0])
                        lib.plot.plot('inception_50k_std', inception_score[1])

                    # Calculate dev loss and generate samples every 100 iters
                    if iteration % 100 == 99:
                        dev_disc_costs = []
                        for images, _labels in dev_gen():
                            _dev_disc_cost = session.run([disc_cost],
                                                         feed_dict={all_real_data_int: images,
                                                                    all_real_labels: _labels})
                            dev_disc_costs.append(_dev_disc_cost)
                        lib.plot.plot('dev_cost', np.mean(dev_disc_costs))

                        g_samples = generate_image(iteration, _data, cifar_image_save_path)

                        global_step.assign(iteration).eval()
                        saver_gen.save(session, ckpt_gen_cifar_path_new + './resnet_model_gen_cifar_ckpt',
                                       global_step=global_step)
                        saver_disc.save(session, ckpt_disc_cifar_path_new + "./resnet_model_disc_cifar_ckpt",
                                        global_step=global_step)
                        test_images_path = cifar_image_save_path + './test{}'.format(anti_label)
                        if not os.path.exists(test_images_path):
                            os.makedirs(test_images_path)

                        lib.save_images.save_images(_data.reshape((64, 3, 32, 32)),
                                                    test_images_path + '/samples_{}.png'.format(cpz))
                        cpz += 1.

                    if (iteration < 200) or (iteration % 100 == 99):
                        lib.plot.flush()

                    lib.plot.tick()

                    if iteration == 0:
                        gen_sample = generate_image(iteration, _data, cifar_image_save_path)
                        print("gen_sample shape:",np.shape(gen_sample))
                        x_value, y_value = lib.cifar10.SVM_DATA_LOAD(anti_label)
                        print(type(gen_sample))
                        gen_sample_array = gen_sample
                        sample_len = len(gen_sample_array)#append处应该是问题所在导致了整体尺寸不对
                        x_value = np.append(x_value,gen_sample,axis=0)
                        print("x_value shape:",np.shape(x_value))
                        sample_targets = np.ones([sample_len,])
                        print(np.shape(sample_targets))
                        print("y_value shape:", np.shape(y_value))
                        y_value = np.append(y_value,sample_targets,axis=0)

                        #pca, train part
                        print("start PCA train data part")
                        pca=PCA(n_components=900,copy=True,whiten=False)
                        pca.fit_transform(x_value)
                        print("x_value shape:", np.shape(x_value))

                        D0_SVM = SVC(class_weight='balanced', kernel='rbf')
                        print("fit start")
                        D0_SVM.fit(x_value, y_value)
                        print("fit over")
                        D0_SVM_PATH = './D0_SVM_PATH/anti_label_{}'.format(anti_label)
                        if not os.path.exists(D0_SVM_PATH):
                            os.makedirs(D0_SVM_PATH)
                        joblib.dump(D0_SVM, D0_SVM_PATH+'./D0_SVM.pkl')

                        test_x_values, test_y_values = lib.cifar10.SVM_TEST_LOAD(anti_label)
                        #pca test data part
                        pca.transform(test_x_values)
                        print("start test predict")
                        test_predict = D0_SVM.predict(test_x_values)
                        print("test predict over")
                        test_len = len(test_predict)
                        print(test_len)
                        test_sum = np.sum(test_len, axis=0)
                        print(test_sum)

                gt_sample = generate_image(iteration + 1, _data, cifar_image_save_path)
                # gt_sample_array = gt_sample.eval()
                pca.transform(gt_sample)
                predict_result = D0_SVM.predict(gt_sample)
                print("predict over")
                fff = open('./predict_result/result.txt', 'a')
                fff.write("minus_value =" + str(minus_value))
                fff.write("   ST_ITERS =" + str(ST_ITERS))
                fff.write('\n')
                fff.write("anti_number is :" + str(anti_label))
                fff.write('\n')
                for zz in range(10):
                    part_result = predict_result[zz * 10:zz * 10 + 10]
                    fff.write(str(part_result))
                    fff.write('\n')
                predict_sum = np.sum(predict_result)
                predict_minus_value = 0.5 - (predict_sum / 200)
                fff.write("predict_minus_value is ")
                fff.write(str(predict_minus_value))
                fff.write('\n')
                fff.write('\n')
                fff.close()

                G_SYMBOL = True
                D_SYMBOL = True
                '''print("start move disc path")
                shutil.copytree("E:\\python_test\\untitled\\resnet_ST_COMBINE\\resnet_disc_cifar_path_new\\anti_label_{}".format(anti_label),
                            "H:\\model test result\\cifar\\automobile\\test\\disc\\anti_label_{}".format(anti_label))
                print("start move gen path")
                shutil.copytree(
                    "E:\\python_test\\untitled\\resnet_ST_COMBINE\\resnet_gen_cifar_path_new\\anti_label_{}".format(
                        anti_label),
                    "H:\\model test result\\cifar\\automobile\\test\\gen\\anti_label_{}".format(anti_label))
                '''

#tensorflow_load()
#tensor_main()

