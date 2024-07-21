import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Layer
from Params import args

'''class AdditiveAttention(Layer):
    def __init__(self, query_vector_dim, candidate_vector_dim, writer=None, tag=None, names=None):
        self.query_vector_dim = query_vector_dim
        self.candidate_vector_dim = candidate_vector_dim
        self.attention_query_vector = tf.Variable(tf.random.uniform(shape=[query_vector_dim, 1], minval=-0.1, maxval=0.1))

    def attention(self, candidate_vector):
        """
        Args:
            candidate_vector: batch_size, candidate_size, candidate_vector_dim
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        with tf.name_scope('additive_attention'):
            dense = tf.keras.layers.Dense(self.query_vector_dim)(candidate_vector)
            temp = tf.tanh(dense)
            candidate_weights = tf.nn.softmax(tf.squeeze(tf.matmul(temp, self.attention_query_vector), axis=2), axis=1)
            target = tf.squeeze(tf.matmul(tf.expand_dims(candidate_weights, 1), candidate_vector), 1)
            return target'''

class ScaledDotProductAttention(Layer):
    def __init__(self, d_k):
        self.d_k = d_k

    def attention(self, Q, K, V, attn_mask=None):
        with tf.name_scope('scaled_attention'): 
            scores = tf.matmul(Q, tf.transpose(K,perm=[0,1,3,2])) / np.sqrt(self.d_k)
            scores = tf.exp(scores)
            if attn_mask is not None:
                scores = scores * attn_mask
            attn = scores / (tf.expand_dims(tf.reduce_sum(scores, axis=-1),-1) + 1e-8)
            context = tf.matmul(attn, V)
            return context, attn
    
    def __call__(self, inputs):
        Q, K, V = inputs  # Assuming inputs are Q, K, V tensors
        return self.attention(Q, K, V)

class MultiHeadSelfAttention(Layer):
    def __init__(self, d_model, num_attention_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads

    def build(self, input_shape):
        self.dense1 = Dense(self.d_model,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=None))
        self.dense2 = Dense(self.d_model,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=None))
        self.dense3 = Dense(self.d_model,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=None))
        pass

    def attention(self, Q, K=None, V=None, length=None):
        """
        Q:batch_size,candidate_num,embedding_size
        return : batch_size,candidate_num,embedding_size
        """
        with tf.name_scope('multihead_selfattention'): 
            if K is None:
                K = Q
            if V is None:
                V = Q
            batch_size = len(Q)

            W_Q = self.dense1(Q)
            q_s = tf.transpose(tf.reshape(W_Q,[batch_size, -1, self.num_attention_heads,self.d_k]),perm=[0,2,1,3])
            W_K = self.dense2(K)
            k_s = tf.transpose(tf.reshape(W_K,[batch_size, -1, self.num_attention_heads,self.d_k]),perm=[0,2,1,3])
            W_V = self.dense3(V)
            v_s = tf.transpose(tf.reshape(W_V,[batch_size, -1, self.num_attention_heads,self.d_v]),perm=[0,2,1,3])
            # batch_size,head_num, candidate_num, d_k
            context, attn = ScaledDotProductAttention(self.d_k)([q_s, k_s, v_s])
            # batch_size,candidate_num,embedding_size
            context= tf.reshape(tf.transpose(context,perm=[0,2,1,3]),[batch_size, -1, self.num_attention_heads*self.d_v])
            return context


    def call(self, inputs):
        return self.attention(inputs)
