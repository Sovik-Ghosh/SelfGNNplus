import tensorflow as tf

# Define a tensor
scores = tf.constant([[1, 2, 3], [4, 5, 6]])

# Compute the sum along the last axis and keep dimensions
result = tf.exp(scores)
result1 = tf.nn.softmax(scores, axis=-1)

print(result, result1)