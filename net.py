# import tensorflow
import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd


# Data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

train = pd.read_csv(os.path.join(BASE_DIR, 'keys.csv'))
values = pd.read_csv(os.path.join(BASE_DIR, 'values.csv'))

val_x = pd.read_csv(os.path.join(BASE_DIR, 'valx.csv'))
val_y = pd.read_csv(os.path.join(BASE_DIR, 'valy.csv'))
# test = pd.read_csv(os.path.join(BASE_DIR, 'Test.csv'))

# -------------------------

seed = 128

input_num_units = 4 # Sin contar tipo de roca
hidden_num_units = 5
output_num_units = 5

# build computational graph
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# other vars
epochs = 5
batch_size = 20
learning_rate = 0.01

# addition = tf.add(a, b)

# Network architecture =============================================================
# define weights and bias
weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}


# create neural networks computational graph
hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)

output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']


# cost of neural network
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))

# usa adam optimizer una version mas eficiente de gradient descent optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Initialize variables
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

# create session and run the graph
with tf.Session() as sess:
    sess.run(init)  # Create initialized variables

    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(train.shape[0] / batch_size)

        # batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train')
        _, c = sess.run([optimizer, cost], feed_dict = {x: train, y: values})  # features, labels

        avg_cost += c / total_batch

        print "Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_cost)

    print "\nTraining complete!"

    # pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    # out = tf.Print(output_layer, [output_layer])
    # oses = tf.InteractiveSession()
    # oses.run(out)

    pred_temp = tf.equal(output_layer, y)

    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))

    # print "Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, input_num_units), y: val_y})
    # val_x = np.asmatrix(val_x)
    # print('VALXLXL')
    # print(val_x)
    # val_x = tf.constant(val_x, dtype = tf.float32, shape=[9, 4])

    # print "Validation Accuracy:", accuracy.eval({ x: val_x.reshape(-1, input_num_units), y: val_y})
    print "Validation Accuracy:", accuracy.eval({ x: val_x, y: val_y})

# close session
sess.close()




