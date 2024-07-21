import tensorflow as tf
tf.config.run_functions_eagerly

# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Access trainable variables
trainable_vars = model.trainable_variables
print(f"Trainable Variables:{trainable_vars}")
for var in trainable_vars:
    print(var.name, var.shape)