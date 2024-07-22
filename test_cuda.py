import tensorflow as tf

# Create sample tensors
float_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
int_tensor = tf.constant([1, 2, 3], dtype=tf.int32)

# Check data types
print("Float Tensor dtype:", float_tensor.dtype)
print("Int Tensor dtype:", int_tensor.dtype)

# Example operation that requires integer tensor
indices = tf.constant([0, 1, 2], dtype=tf.int32)
values = tf.constant([10.0, 20.0, 30.0], dtype=tf.float32)
result = tf.gather(values, indices)

print("Gather result:", result)