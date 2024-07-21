import tensorflow as tf

def stopgrad(x):
    return tf.stop_gradient(x)

def do_slice(x, a, b):
    return tf.slice(x, a, b)

def shape(x):
    return tf.shape(x)

def leaky_relu(x):
    return tf.nn.leaky_relu(x)

def embedded_lookup(x, y):
    return tf.nn.embedding_lookup(x, y)