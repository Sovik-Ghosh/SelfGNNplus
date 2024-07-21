import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import Layer
import numpy as np

import warnings

# Suppress all warnings
warnings.filterwarnings('ignore', category=FutureWarning)

paramId = 0
biasDefault = False
params = {}
regParams = {}
ita = 0.2
leaky = 0.1

def getParamId():
    global paramId
    paramId += 1
    return paramId

def setIta(ITA):
    global ita
    ita = ITA

def setBiasDefault(val):
    global biasDefault
    biasDefault = val

def getParam(name):
    return params[name]

def addReg(name, param):
    global regParams
    if name not in regParams:
        regParams[name] = param
    else:
        print('ERROR: Parameter already exists')

def addParam(name, param):
    global params
    if name not in params:
        params[name] = param

def defineRandomNameParam(shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
    name = 'defaultParamName%d' % getParamId()
    return defineParam(name, shape, dtype, reg, initializer, trainable)

def defineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
    global params
    global regParams
    assert name not in params, 'name %s already exists' % name
    if initializer == 'xavier':
        initializer_fn = GlorotUniform()
    elif initializer == 'trunc_normal':
        initializer_fn = tf.random.truncated_normal_initializer(mean=0.0, stddev=0.03)
    elif initializer == 'zeros':
        initializer_fn = tf.zeros_initializer()
    elif initializer == 'ones':
        initializer_fn = tf.ones_initializer()
    else:
        print('ERROR: Unrecognized initializer')
        exit()

    ret = tf.Variable(initializer_fn(shape=shape, dtype=dtype), trainable=trainable, name=name)
    
    params[name] = ret
    if reg:
        regParams[name] = ret
    return ret

def getOrDefineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True, reuse=False):
    global params
    global regParams
    if name in params:
        assert reuse, 'Reusing Param %s Not Specified' % name
        if reg and name not in regParams:
            regParams[name] = params[name]
        return params[name]
    return defineParam(name, shape, dtype, reg, initializer, trainable)

def BN(inp, name=None):
    global ita
    dim = inp.shape[-1]
    name = 'defaultParamName%d' % getParamId()
    scale = tf.Variable(tf.ones([dim]), name=name + '_scale')
    shift = tf.Variable(tf.zeros([dim]), name=name + '_shift')
    mean, variance = tf.nn.moments(inp, axes=[0])
    ret = tf.nn.batch_normalization(inp, mean, variance, shift, scale, 1e-8)
    return ret


def FC(inp, outDim, name=None, useBias=False, activation=None, reg=False, useBN=False, dropout=None, initializer='xavier', reuse=False, biasReg=False, biasInitializer='zeros'):
    global params
    global regParams
    global leaky
    inDim = inp.shape[-1]
    temName = name if name != None else 'defaultParamName%d' % getParamId()
    W = getOrDefineParam(temName, [inDim, outDim], reg=reg, initializer=initializer, reuse=reuse)
    if dropout is not None:
        ret = tf.nn.dropout(inp, rate=dropout) @ W
    else:
        ret = inp @ W
    if useBias:
        ret = Bias(ret, name=name, reuse=reuse, reg=biasReg, initializer=biasInitializer)
    if useBN:
        ret = BN(ret)
    if activation is not None:
        ret = Activate(ret, activation)
    return ret

def Bias(data, name=None, reg=False, reuse=False, initializer='zeros'):
    inDim = data.shape[-1]
    temName = name if name != None else 'defaultParamName%d' % getParamId()
    temBiasName = temName + 'Bias'
    bias = getOrDefineParam(temBiasName, [inDim], reg=False, initializer=initializer, reuse=reuse)
    if reg:
        regParams[temBiasName] = bias
    return data + bias

def ActivateHelp(data, method):
    if method == 'relu':
        ret = tf.nn.relu(data)
    elif method == 'sigmoid':
        ret = tf.nn.sigmoid(data)
    elif method == 'tanh':
        ret = tf.nn.tanh(data)
    elif method == 'softmax':
        ret = tf.nn.softmax(data, axis=-1)
    elif method == 'leakyRelu':
        ret = tf.maximum(data, leaky * data)
    elif method == 'twoWayLeakyRelu6':
        temMask = tf.cast(tf.greater(data, 6.0), tf.float32)
        ret = temMask * (6 + leaky * (data - 6)) + (1 - temMask) * tf.maximum(leaky * data, data)
    elif method == '-1relu':
        ret = tf.maximum(-1.0, data)
    elif method == 'relu6':
        ret = tf.maximum(0.0, tf.minimum(6.0, data))
    elif method == 'relu3':
        ret = tf.maximum(0.0, tf.minimum(3.0, data))
    else:
        raise Exception('Error Activation Function')
    return ret

def Activate(data, method, useBN=False):
    global leaky
    if useBN:
        ret = BN(data)
    else:
        ret = data
    ret = ActivateHelp(ret, method)
    return ret

def Regularize(names=None, method='L2'):
    ret = 0
    if method == 'L1':
        if names is not None:
            for name in names:
                ret += tf.reduce_sum(tf.abs(getParam(name)))
        else:
            for name in regParams:
                ret += tf.reduce_sum(tf.abs(regParams[name]))
    elif method == 'L2':
        if names is not None:
            for name in names:
                ret += tf.reduce_sum(tf.square(getParam(name)))
        else:
            for name in regParams:
                ret += tf.reduce_sum(tf.square(regParams[name]))
    return ret

def Dropout(data, rate):
    if rate is None:
        return data
    else:
        return tf.nn.dropout(data, rate=rate)

def selfAttention(localReps, number, inpDim, numHeads):
    Q = defineRandomNameParam([inpDim, inpDim], reg=True)
    K = defineRandomNameParam([inpDim, inpDim], reg=True)
    V = defineRandomNameParam([inpDim, inpDim], reg=True)
    rspReps = tf.reshape(tf.stack(localReps, axis=1), [-1, inpDim])
    q = tf.reshape(rspReps @ Q, [-1, number, 1, numHeads, inpDim//numHeads])
    k = tf.reshape(rspReps @ K, [-1, 1, number, numHeads, inpDim//numHeads])
    v = tf.reshape(rspReps @ V, [-1, 1, number, numHeads, inpDim//numHeads])
    att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True) / tf.sqrt(inpDim/numHeads), axis=2)
    attval = tf.reshape(tf.reduce_sum(att * v, axis=2), [-1, number, inpDim])
    rets = [None] * number
    paramId = 'dfltP%d' % getParamId()
    for i in range(number):
        tem1 = tf.reshape(tf.slice(attval, [0, i, 0], [-1, 1, -1]), [-1, inpDim])
        rets[i] = tem1 + localReps[i]
    return rets

def lightSelfAttention(localReps, number, inpDim, numHeads):
    Q = defineRandomNameParam([inpDim, inpDim], reg=True)
    rspReps = tf.reshape(tf.stack(localReps, axis=1), [-1, inpDim])
    tem = rspReps @ Q
    q = tf.reshape(tem, [-1, number, 1, numHeads, inpDim//numHeads])
    k = tf.reshape(tem, [-1, 1, number, numHeads, inpDim//numHeads])
    v = tf.reshape(rspReps, [-1, 1, number, numHeads, inpDim//numHeads])
    att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True) / tf.sqrt(inpDim/numHeads), axis=2)
    attval = tf.reshape(tf.reduce_sum(att * v, axis=2), [-1, number, inpDim])
    rets = [None] * number
    paramId = 'dfltP%d' % getParamId()
    for i in range(number):
        tem1 = tf.reshape(tf.slice(attval, [0, i, 0], [-1, 1, -1]), [-1, inpDim])
        rets[i] = tem1 + localReps[i]
    return rets

class LSTMNet(tf.keras.layers.Layer):
    def __init__(self, layers=1, hidden_units=100, hidden_activation="tanh", dropout=0.5):
        super(LSTMNet, self).__init__()
        self.layers = layers
        self.hidden_units = hidden_units
        self.hidden_activation = self.get_activation_function(hidden_activation)
        self.dropout = dropout

    def get_activation_function(self, activation):
        if activation == "tanh":
            return tf.nn.tanh
        elif activation == "relu":
            return tf.nn.relu
        else:
            raise NotImplementedError(f"Activation function '{activation}' is not implemented")

    def build(self, input_shape):
        # This method is where you create the LSTM cells and weights, if necessary
        self.cells = [tf.keras.layers.LSTMCell(self.hidden_units, activation=self.hidden_activation, dropout = self.dropout) for _ in range(self.layers)]
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(self.cells), return_sequences=True)

    def call(self, inputs, training=False):
        outputs = self.rnn(inputs, training=training)
        return outputs

class TemporalConvNet(object):
    def __init__(self, num_channels, stride=1, kernel_size=2, dropout=0.2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_levels = len(num_channels)
        self.num_channels = num_channels
        self.dropout = dropout
    
    def __call__(self, inputs, mask):
        inputs_shape = inputs.shape
        outputs = [inputs]
        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_channels = inputs_shape[-1] if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            output = self._TemporalBlock(outputs[-1], in_channels, out_channels, self.kernel_size, 
                                    self.stride, dilation=dilation_size, padding=(self.kernel_size-1)*dilation_size, 
                                    dropout=self.dropout, level=i)
            outputs.append(output)

        return outputs[-1]

    def _TemporalBlock(self, value, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, level=0):
        padded_value1 = tf.pad(value, [[0,0], [padding,0], [0,0]])
        self.conv1 = tf.keras.layers.Conv1D(filters=n_outputs,
                                            kernel_size=kernel_size,
                                            strides=stride,
                                            padding='valid',
                                            dilation_rate=dilation,
                                            activation=None,
                                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                            bias_initializer=tf.zeros_initializer(),
                                            name='layer'+str(level)+'_conv1')(padded_value1)
        self.output1 = tf.nn.dropout(tf.nn.relu(self.conv1), rate=dropout)

        padded_value2 = tf.pad(self.output1, [[0,0], [padding,0], [0,0]])
        self.conv2 = tf.keras.layers.Conv1D(filters=n_outputs,
                                            kernel_size=kernel_size,
                                            strides=stride,
                                            padding='valid',
                                            dilation_rate=dilation,
                                            activation=None,
                                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                            bias_initializer=tf.zeros_initializer(),
                                            name='layer'+str(level)+'_conv2')(padded_value2)
        self.output2 = tf.nn.dropout(tf.nn.relu(self.conv2), rate=dropout)

        if n_inputs != n_outputs:
            res_x = tf.keras.layers.Conv1D(filters=n_outputs,
                                           kernel_size=1,
                                           activation=None,
                                           kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                           bias_initializer=tf.zeros_initializer(),
                                           name='layer'+str(level)+'_conv')(value)
        else:
            res_x = value
        return tf.nn.relu(res_x + self.output2)