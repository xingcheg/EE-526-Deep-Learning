from __future__ import print_function
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


# Parameters
learning_rate = 1e-3
niter = 10000
batch_size = 500
display_step = 100


# Network Parameters
n_hidden_1 = 50  # 1st layer number of neurons
n_hidden_2 = 50  # 2nd layer number of neurons
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)


# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])
keep_prob = tf.placeholder(tf.float32)


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.truncated_normal([num_input, n_hidden_1], stddev=0.01)),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.truncated_normal([n_hidden_2, num_classes], stddev=0.01))
}
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.01)),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.truncated_normal([num_classes], stddev=0.01))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 'n_hidden_1' neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    # Hidden fully connected layer with 'n_hidden_2' neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)


# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, niter+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST training and testing images

    print("Training Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.train.images,
                                      Y: mnist.train.labels,
                                      keep_prob: 1.0}))

    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels,
                                      keep_prob: 1.0}))

