import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization
import numpy as np
from Params import args

import warnings

# Suppress all warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class ScaledDotProductAttention(Layer):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def call(self, Q, K, V, attn_mask=None):
        scores = tf.matmul(Q, tf.transpose(K, perm=[0, 1, 3, 2])) / tf.sqrt(tf.cast(self.d_k, tf.float32))
        scores = tf.nn.softmax(scores, axis=-1)
        if attn_mask is not None:
            scores *= attn_mask
        attn = scores / (tf.reduce_sum(scores, axis=-1, keepdims=True) + 1e-8)
        context = tf.matmul(attn, V)
        return context, attn

class AdditiveAttention(tf.keras.layers.Layer):
    def __init__(self, query_vector_dim = args.query_vector_dim, candidate_vector_dim = args.latdim, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.query_vector_dim = query_vector_dim
        self.candidate_vector_dim = candidate_vector_dim
        self.attention_query_vector = None  # To be initialized in build()

    def build(self, input_shape):
        # Initialize the attention query vector with the appropriate shape
        self.attention_query_vector = self.add_weight(
            shape=(self.query_vector_dim,),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
            trainable=True,
            name='attention_query_vector'
        )
        super(AdditiveAttention, self).build(input_shape)

    def call(self, candidate_vector, training=False):
        """
        Args:
            candidate_vector: Tensor with shape (batch_size, candidate_size, candidate_vector_dim)
        Returns:
            Tensor with shape (batch_size, candidate_vector_dim)
        """
        with tf.name_scope('additive_attention'):
            # Apply a Dense layer to candidate_vector
            dense = tf.keras.layers.Dense(self.query_vector_dim)(candidate_vector)
            temp = tf.tanh(dense)
            
            # Compute attention scores
            scores = tf.squeeze(tf.matmul(temp, tf.expand_dims(self.attention_query_vector, axis=-1)), axis=-1)
            attention_weights = tf.nn.softmax(scores, axis=1)
            
            # Reshape attention_weights to match candidate_vector for element-wise multiplication
            attention_weights = tf.expand_dims(attention_weights, axis=-1)  # Shape: (batch_size, candidate_size, 1)
            
            # Apply attention weights to candidate_vector to get the context vector
            context_vector = tf.reduce_sum(attention_weights * candidate_vector, axis=1)
            return context_vector

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.query_vector_dim)

    def get_config(self):
        config = super(AdditiveAttention, self).get_config()
        config.update({
            'query_vector_dim': self.query_vector_dim,
            'candidate_vector_dim': self.candidate_vector_dim
        })
        return config


class MultiHeadSelfAttention(Layer):
    def __init__(self, d_model, num_attention_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads
        self.dense_q = Dense(d_model)
        self.dense_k = Dense(d_model)
        self.dense_v = Dense(d_model)
        self.dense_o = Dense(d_model)

    def call(self, Q, K=None, V=None, mask=None):
        if K is None:
            K = Q
        if V is None:
            V = Q

        batch_size = tf.shape(Q)[0]
        Q = self.dense_q(Q)
        K = self.dense_k(K)
        V = self.dense_v(V)

        Q = tf.reshape(Q, (batch_size, -1, self.num_attention_heads, self.d_k))
        K = tf.reshape(K, (batch_size, -1, self.num_attention_heads, self.d_k))
        V = tf.reshape(V, (batch_size, -1, self.num_attention_heads, self.d_v))

        Q = tf.transpose(Q, perm=[0, 2, 1, 3])
        K = tf.transpose(K, perm=[0, 2, 1, 3])
        V = tf.transpose(V, perm=[0, 2, 1, 3])

        context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V, mask)

        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, self.num_attention_heads * self.d_v))

        return self.dense_o(context)

class FeedForward(Layer):
    def __init__(self, num_units, dropout_keep_prob=1.0):
        super(FeedForward, self).__init__()
        self.num_units = num_units
        self.dense1 = Dense(num_units[0], activation=tf.nn.relu)
        self.dense2 = Dense(num_units[1])
        self.dropout = Dropout(1.0 - dropout_keep_prob)

    def call(self, inputs):
        outputs = self.dense1(inputs)
        outputs = self.dropout(outputs)
        outputs = self.dense2(outputs)
        outputs = self.dropout(outputs)
        outputs += inputs  # Residual connection
        return outputs

class TransformerNet(tf.keras.layers.Layer):
    def __init__(self, num_units, num_blocks, num_heads, maxlen, dropout_rate, pos_fixed, l2_reg=0.0):
        super(TransformerNet, self).__init__()
        self.num_units = num_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.maxlen = maxlen
        self.dropout_keep_prob = 1. - dropout_rate
        self.pos_fixed = pos_fixed
        self.l2_reg = l2_reg
        self.position_encoding_matrix = None
        
        self.attention_layers = [MultiHeadSelfAttention(num_units, num_heads) for _ in range(num_blocks)]
        self.feedforward_layers = [FeedForward(num_units=[num_units, num_units], dropout_keep_prob=self.dropout_keep_prob) for _ in range(num_blocks)]
        self.norm_layers = [tf.keras.layers.LayerNormalization() for _ in range(num_blocks * 2)]

    def build(self, input_shape):
        self.pos_embedding_table = self.add_weight(
            name='pos_embedding_tnet', 
            shape=[self.maxlen, self.num_units], 
            dtype=tf.float32, 
            regularizer=tf.keras.regularizers.l2(self.l2_reg),
            initializer='glorot_uniform',
            trainable = True
        )
        super(TransformerNet, self).build(input_shape)

    def position_embedding(self, inputs):
        # Use the pre-defined positional embedding table
        outputs = tf.nn.embedding_lookup(self.pos_embedding_table, inputs)
        return outputs

    def get_position_encoding(self, inputs):
        # If pos_fixed is True, use a predefined position encoding matrix
        if self.position_encoding_matrix is None:
            encoded_vec = np.array([pos / np.power(10000, 2 * i / self.num_units) for pos in range(self.maxlen) for i in range(self.num_units)])
            encoded_vec[::2] = np.sin(encoded_vec[::2])
            encoded_vec[1::2] = np.cos(encoded_vec[1::2])
            encoded_vec = tf.convert_to_tensor(encoded_vec.reshape([self.maxlen, self.num_units]), dtype=tf.float32)
            self.position_encoding_matrix = encoded_vec
        
        N = tf.shape(inputs)[0]
        T = tf.shape(inputs)[1]
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (batch_size, len)
        position_encoding = tf.nn.embedding_lookup(self.position_encoding_matrix, position_ind)  # (batch_size, len, num_units)
        return position_encoding

    def call(self, inputs, training=False):
        if self.pos_fixed:  # use sin/cos positional encoding
            position_encoding = self.get_position_encoding(inputs)  # (batch_size, len, num_units)
        else:
            position_encoding = self.position_embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(inputs)[1]), 0), [tf.shape(inputs)[0], 1])
            )
        
        inputs += position_encoding

        for i in range(self.num_blocks):
            inputs = self.norm_layers[2 * i](inputs, training=training)
            inputs = self.attention_layers[i](inputs)
            inputs = self.norm_layers[2 * i + 1](inputs, training=training)
            inputs = self.feedforward_layers[i](inputs, training=training)

        outputs = self.norm_layers[-1](inputs, training=training)
        return outputs


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

class GRUNet(tf.keras.layers.Layer):
    def __init__(self, layers=1, hidden_units=100, hidden_activation="tanh", dropout=0.5):
        super(GRUNet, self).__init__()
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
        self.cells = [tf.keras.layers.GRUCell(self.hidden_units, activation=self.hidden_activation, dropout = self.dropout) for _ in range(self.layers)]
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(self.cells), return_sequences=True)

    def call(self, inputs, training=False):
        outputs = self.rnn(inputs, training=training)
        return outputs

class TemporalConvNet(tf.keras.layers.Layer):
    def __init__(self, num_channels, stride=1, kernel_size=2, dropout=0.2, **kwargs):
        super(TemporalConvNet, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_levels = len(num_channels)
        self.num_channels = num_channels
        self.dropout = dropout
    
    def build(self, input_shape):
        # Build convolutional layers here
        self.convs = []
        self.dropout_layers = []
        self.residual_convs = []

        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_channels = input_shape[-1] if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            
            conv1 = tf.keras.layers.Conv1D(filters=out_channels,
                                           kernel_size=self.kernel_size,
                                           strides=self.stride,
                                           padding='valid',
                                           dilation_rate=dilation_size,
                                           activation=None,
                                           name='layer{}_conv1'.format(i))
            
            conv2 = tf.keras.layers.Conv1D(filters=out_channels,
                                           kernel_size=self.kernel_size,
                                           strides=self.stride,
                                           padding='valid',
                                           dilation_rate=dilation_size,
                                           activation=None,
                                           name='layer{}_conv2'.format(i))
            
            dropout_layer = tf.keras.layers.Dropout(rate=self.dropout)
            self.convs.append((conv1, conv2))
            self.dropout_layers.append(dropout_layer)
            
            if in_channels != out_channels:
                res_conv = tf.keras.layers.Conv1D(filters=out_channels,
                                                  kernel_size=1,
                                                  activation=None,
                                                  name='layer{}_conv'.format(i))
                self.residual_convs.append(res_conv)
            else:
                self.residual_convs.append(None)
    
    def call(self, inputs, training=False):
        x = inputs
        for i in range(self.num_levels):
            conv1, conv2 = self.convs[i]
            dropout_layer = self.dropout_layers[i]
            res_conv = self.residual_convs[i]

            # Apply first convolution
            x = tf.pad(x, [[0,0], [(self.kernel_size-1)*(2**i), 0], [0, 0]])
            x = conv1(x)
            x = dropout_layer(tf.nn.relu(x), training=training)

            # Apply second convolution
            x = tf.pad(x, [[0,0], [(self.kernel_size-1)*(2**i), 0], [0, 0]])
            x = conv2(x)
            x = dropout_layer(tf.nn.relu(x), training=training)

            # Apply residual connection
            if res_conv is not None:
                res_x = res_conv(inputs)
            else:
                res_x = inputs

            x = tf.nn.relu(res_x + x)
            inputs = x  # Update inputs for the next layer
        
        return x