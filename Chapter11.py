import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

#%% Vanishing/Exploding Gradients Problem
def logit(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-5, 5, 200)
plt.figure()
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [1.0, 1.0], 'k--')
plt.plot([0, 0], [-0.2, 1.2], 'k-')
plt.plot([-5, 5], [-3/4, 7/4], 'g--')
plt.plot(z, logit(z), 'b-', linewidth=2)
props = dict(facecolor='black', shrink=0.1)
plt.annotate('Saturating', xytext=(3.5, 0.7), xy=(5, 1), arrowprops=props, fontsize=14, ha="center")
plt.annotate('Saturating', xytext=(-3.5, 0.3), xy=(-5, 0), arrowprops=props, fontsize=14, ha="center")
plt.annotate('Linear', xytext=(2, 0.2), xy=(0, 0.5), arrowprops=props, fontsize=14, ha="center")
plt.grid(True)
plt.title('Sigmoid activation function', fontsize=14)
plt.axis([-5, 5, -0.2, 1.2])

#%% Xavier and He Initialization
reset_graph()

n_inputs = 28 * 28  #MNIST
n_hidden1 = 300

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
he_init = tf.variance_scaling_initializer()
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
                          kernel_initializer=he_init, name='hidden1')

#%% Nonsaturating Activation Functions
# Leaky ReLu
def leaky_relu(z, alpha=0.01):
    return np.maximum(alpha * z, z)

z = np.linspace(-5, 5, 200)
plt.figure()
plt.plot(z, leaky_relu(z, 0.05), 'b-', linewidth=2)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([0, 0], [-0.5, 4.2], 'k-')
plt.grid(True)
props = dict(facecolor='black', shrink=0.1)
plt.annotate('Leak', xytext=(-3.5, 0.5), xy=(-5, -0.2), arrowprops=props, fontsize=14, ha="center")
plt.axis([-5, 5, -0.5, 4.2])
plt.title('Leaky ReLU activation function', fontsize=14)

#%% Leaky ReLU in tensorflow
reset_graph()

def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None), name='y')
he_init = tf.variance_scaling_initializer()

with tf.name_scope('dnn'):
    #hidden1 = tf.layers.dense(X, n_hidden1, activation=leaky_relu,
    #                      kernel_initializer=he_init, name='hiddne1')
    #hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=leaky_relu,
    #                      kernel_initializer=he_init, name='hidden2')
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.elu,
                          kernel_initializer=he_init, name='hiddne1')
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.elu,
                          kernel_initializer=he_init, name='hidden2')
    logits = tf.layers.dense(hidden2, n_outputs, name='outputs')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

learning_rate = 0.01
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
       
n_epochs = 40
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if epoch % 5 == 0:
            acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, 'Batch accuracy:', acc_batch, 'Validation accuracy:', acc_valid)

#%% ELU
def elu(z, alpha=1):
    return np.where(z < 0, alpha * (np.exp(z) - 1), z)

z = np.linspace(-5, 5, 200)
plt.figure()
plt.plot(z, elu(z), 'b-', linewidth=2)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [-1, -1], 'k--')
plt.plot([0, 0], [-2.2, 3.2], 'k-')
plt.grid(True)
plt.title('ELU activation function', fontsize=14)
plt.axis([-5, 5, -2.2, 3.2])

#%% SELU
def selu(z, scale=1.050700987355480493419, alpha=1.673263242354377284817):
    return scale * elu(z, alpha)

z = np.linspace(-5, 5, 200)
plt.figure()
plt.plot(z, selu(z), 'b-', linewidth=2)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [-1.758, -1.758], 'k--')
plt.plot([0, 0], [-2.2, 3.2], 'k-')
plt.grid(True)
plt.title('SELU activation function', fontsize=14)
plt.axis([-5, 5, -2.2, 3.2])

#%% SELU in tensorflow
reset_graph()

def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None), name='y')
he_init = tf.variance_scaling_initializer()

with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.selu,
                          kernel_initializer=he_init, name='hiddne1')
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.selu,
                          kernel_initializer=he_init, name='hidden2')
    logits = tf.layers.dense(hidden2, n_outputs, name='outputs')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

learning_rate = 0.01
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]
        
n_epochs = 40
batch_size = 50

means = X_train.mean(axis=0, keepdims=True)
stds = X_train.std(axis=0, keepdims=True) + 1e-10
X_val_scaled = (X_valid - means) / stds

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            X_batch_scaled = (X_batch - means) / stds
            sess.run(training_op, feed_dict={X: X_batch_scaled, y: y_batch})
        if epoch % 5 == 0:
            acc_batch = accuracy.eval(feed_dict={X: X_batch_scaled, y: y_batch})
            acc_valid = accuracy.eval(feed_dict={X: X_val_scaled, y: y_valid})
            print(epoch, 'Batch accuracy:', acc_batch, 'Validation accuracy:', acc_valid)

#%% Batch Normalization
from functools import partial

reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None), name='y')
training = tf.placeholder_with_default(False, shape=(), name='training')

learning_rate = 0.01
batch_norm_momentum = 0.999

with tf.name_scope('dnn'):
    he_init = tf.variance_scaling_initializer()
    
    my_batch_norm_layer = partial(tf.layers.batch_normalization,
                                  training=training,
                                  momentum=batch_norm_momentum)
    my_dense_layer = partial(tf.layers.dense,
                             kernel_initializer=he_init)
    
    hidden1 = my_dense_layer(X, n_hidden1, name='hidden1')
    bn1 = tf.nn.elu(my_batch_norm_layer(hidden1))
    hidden2 = my_dense_layer(bn1, n_hidden2, name='hidden2')
    bn2 = tf.nn.elu(my_batch_norm_layer(hidden2))
    logits_before_bn = my_dense_layer(bn2, n_outputs, name='outputs')
    logits = my_batch_norm_layer(logits_before_bn)
    
with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')
    
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        training_op = optimizer.minimize(loss)
    
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
init = tf.global_variables_initializer()
n_epochs = 30
batch_size = 200
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={training:True, X:X_batch, y:y_batch})
        accuracy_val = accuracy.eval(feed_dict={X:X_valid, y:y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

#%% Gradient Clipping
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 50
n_hidden3 = 50
n_hidden4 = 50
n_hidden5 = 50
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None), name='y')

with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.elu, name='hidden1')
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.elu, name='hidden2')
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.elu, name='hidden3')
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.elu, name='hidden4')
    hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.elu, name='hidden5')
    logits = tf.layers.dense(hidden5, n_outputs, name='outputs')
    
with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')
    
learning_rate = 0.01
threshold = 1.0
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var) for grad, var in grads_and_vars]
    training_op = optimizer.apply_gradients(capped_gvs)
    
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 200
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        accuracy_val = accuracy.eval(feed_dict={X:X_valid, y:y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)
    
    # Another approach is to create a collection containing all the important 
    # operations that people will want to get a handle on:
    for op in (X, y, accuracy, training_op):
        tf.add_to_collection('my_important_ops', op)
    
    save_path = saver.save(sess, "Graph_Check_Points//my_model_final.ckpt")

#%% Reusing a Tensorflow Model
reset_graph()

saver = tf.train.import_meta_graph('Graph_Check_Points/my_model_final.ckpt.meta')

'''
for op in tf.get_default_graph().get_operations():
    print(op.name)

# Once you know which operations you need, you can get a handle on them using
# the graph's get_operation_by_name() or get_tensor_by_name() methods:

X = tf.get_default_graph().get_tensor_by_name('X:0')
y = tf.get_default_graph().get_tensor_by_name('y:0')

accuracy = tf.get_default_graph().get_tensor_by_name('eval/accuracy:0')
training_op = tf.get_default_graph().get_operation_by_name('GradientDescent')
'''

X, y, accuracy, training_op = tf.get_collection('my_important_ops')

n_epochs = 20
batch_size = 200
with tf.Session() as sess:
    saver.restore(sess, 'Graph_Check_Points/my_model_final.ckpt')
    
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, 'Graph_Check_Points/my_model_final.ckpt')

#%% Reuse only the lower layers
reset_graph()

n_hidden4 = 20
n_outputs = 10

saver = tf.train.import_meta_graph('Graph_Check_Points/my_model_final.ckpt.meta')

X = tf.get_default_graph().get_tensor_by_name('X:0')
y = tf.get_default_graph().get_tensor_by_name('y:0')

hidden3 = tf.get_default_graph().get_tensor_by_name('dnn/hidden3/Elu:0')

new_hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name='new_hidden4')
new_logits = tf.layers.dense(new_hidden4, n_outputs, name='new_outputs')

learning_rate = 0.01
with tf.name_scope("new_loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=new_logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("new_eval"):
    correct = tf.nn.in_top_k(new_logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

with tf.name_scope("new_train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
new_saver = tf.train.Saver()

n_epochs = 20
batch_size = 200
with tf.Session() as sess:
    init.run()
    saver.restore(sess, 'Graph_Check_Points/my_model_final.ckpt')

    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = new_saver.save(sess, 'Graph_Check_Points/my_new_model_final.ckpt')
    
#%% Freezing the Lower Layers
# Train a graph
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300 # reused
n_hidden2 = 50  # reused
n_hidden3 = 50  # reused
n_hidden4 = 20  # new!
n_outputs = 10  # new!

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None), name='y')

learning_rate = 0.01
with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name='hidden1')       # reused
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name='hidden2') # reused
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name='hidden3') # reused
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name='hidden4') # new!
    logits = tf.layers.dense(hidden4, n_outputs, name='outputs')                         # new!

with tf.variable_scope('hidden2', reuse=True):
    w_hidden2 = tf.get_variable('kernel')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='hidden[34]|outputs')
    #training_op = optimizer.minimize(loss, var_list=train_vars)
    training_op = optimizer.minimize(loss)
    
reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hidden[123]')  # regular expression
restore_saver = tf.train.Saver(reuse_vars) # to restore layers 1-3

init = tf.global_variables_initializer()

n_epochs = 20
batch_size = 200
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, 'Validation accuracy:', accuracy_val)
    print(w_hidden2.eval())
    restore_saver.save(sess, 'Graph_Check_Points/my_model_final.ckpt')
    
# Use pre-trained graph
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300 # reused
n_hidden2 = 50  # reused
n_hidden3 = 50  # reused
n_hidden4 = 20  # new!
n_outputs = 10  # new!

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None), name='y')

learning_rate = 0.01
with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name='hidden1')       # reused
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name='hidden2') # reused
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name='hidden3') # reused
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name='hidden4') # new!
    logits = tf.layers.dense(hidden4, n_outputs, name='outputs')                         # new!

with tf.variable_scope('hidden2', reuse=True):
    w_hidden2 = tf.get_variable('kernel')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='hidden[34]|outputs')
    training_op = optimizer.minimize(loss, var_list=train_vars)

reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hidden[123]')  # regular expression
restore_saver = tf.train.Saver(reuse_vars) # to restore layers 1-3

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 200
with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, 'Graph_Check_Points/my_model_final.ckpt')
    print(w_hidden2.eval())
    
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, 'Validation accuracy:', accuracy_val)
    save_path = saver.save(sess, 'Graph_Check_Points/my_new_model_final.ckpt')

#%% Caching the Frozen Layers
# Speedup the train process
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300 # reused
n_hidden2 = 50  # reused
n_hidden3 = 50  # reused
n_hidden4 = 20  # new!
n_outputs = 10  # new!

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None), name='y')

learning_rate = 0.01
with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name='hidden1')       # reused
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name='hidden2') # reused
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name='hidden3') # reused
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name='hidden4') # new!
    logits = tf.layers.dense(hidden4, n_outputs, name='outputs')                         # new!

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='hidden[34]|outputs')
    training_op = optimizer.minimize(loss, var_list=train_vars)

reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hidden[123]')  # regular expression
restore_saver = tf.train.Saver(reuse_vars) # to restore layers 1-3

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 200
n_batches = len(X_train) // batch_size

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, 'Graph_Check_Points/my_model_final.ckpt')
    
    h2_cache = sess.run(hidden2, feed_dict={X: X_train})
    h2_cache_valid = sess.run(hidden2, feed_dict={X: X_valid}) # not shown in the book
    
    for epoch in range(n_epochs):
        shuffled_idx = np.random.permutation(len(X_train))
        hidden2_batches = np.array_split(h2_cache[shuffled_idx], n_batches)
        y_batches = np.array_split(y_train[shuffled_idx], n_batches)
        for hidden2_batch, y_batch in zip(hidden2_batches, y_batches):
            sess.run(training_op, feed_dict={hidden2:hidden2_batch, y:y_batch})

        accuracy_val = accuracy.eval(feed_dict={hidden2: h2_cache_valid, # not shown
                                                y: y_valid})             # not shown
        print(epoch, "Validation accuracy:", accuracy_val)               # not shown
    save_path = saver.save(sess, 'Graph_Check_Points/my_new_model_final.ckpt')

#%% Avoiding Overfitting Through Regularization
# l1 and l2 regularization
from functools import partial
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300 # reused
n_hidden2 = 50  # reused
n_outputs = 10  # new!

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None), name='y')

scale = 0.001
my_dense_layer = partial(tf.layers.dense,
                         activation=tf.nn.relu,
                         kernel_regularizer=tf.contrib.layers.l1_regularizer(scale))

learning_rate = 0.01
with tf.name_scope('dnn'):
    hidden1 = my_dense_layer(X, n_hidden1, name='hidden1')
    hidden2 = my_dense_layer(hidden1, n_hidden2, name='hidden2')
    logits = my_dense_layer(hidden2, n_outputs, activation=None, name='outputs')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    base_loss = tf.reduce_mean(xentropy, name='avg_xentropy')
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss] + reg_loss, name='loss')

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 200
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, 'Validation accuracy:', accuracy_val)

    save_path = saver.save(sess, 'Graph_Check_Points/my_new_model_final.ckpt')

#%% Dropout
from functools import partial
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300 # reused
n_hidden2 = 50  # reused
n_outputs = 10  # new!

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None), name='y')
training = tf.placeholder_with_default(False, shape=(), name='training')
dropout_rate = 0.5  # == 1 - keep_prob
X_drop = tf.layers.dropout(X, dropout_rate, training=training)

with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(X_drop, n_hidden1, activation=tf.nn.relu, name='hidden1')
    hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
    hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, activation=tf.nn.relu, name='hidden2')
    hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
    logits = tf.layers.dense(hidden2_drop, n_outputs, name='outputs')
    
with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

with tf.name_scope('train'):
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    training_op = optimizer.minimize(loss)    

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid, training: False})
        print(epoch, 'Validation accuracy:', accuracy_val)

    save_path = saver.save(sess, 'Graph_Check_Points/my_new_model_final.ckpt')
    
#%% Exercise 8
n_inputs = 28 * 28 # MNIST
n_outputs = 5
batch_norm_momentum = 0.98
n_neurons = 160

reset_graph()

he_init = tf.variance_scaling_initializer()
training = tf.placeholder_with_default(False, shape=(), name='training')

def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

def dnn(inputs, n_hidden_layers=5, n_neurons=100, name=None,
        activation=tf.nn.elu, initializer=he_init,
        batch_norm_momentum = 0.999, training=training):
    with tf.variable_scope(name, default_name='dnn'):
        for layer in range(n_hidden_layers):
            inputs = tf.layers.dense(inputs, n_neurons, 
                                     kernel_initializer=initializer,
                                     name='hidden%d' % (layer + 1))
            if batch_norm_momentum != 0:
                inputs = tf.layers.batch_normalization(inputs,
                                                       momentum=batch_norm_momentum,
                                                       training=training)
            inputs = activation(inputs, name="hidden%d_out" % (layer + 1))
        return inputs

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None), name='y')
dnn_outputs = dnn(X, n_neurons=n_neurons,
                  batch_norm_momentum=batch_norm_momentum,
                  activation=tf.nn.relu, training=training)

logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=he_init, name='logits')
Y_proba = tf.nn.softmax(logits, name='Y_proba')

learning_rate = 0.01

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name='loss')

optimizer = tf.train.AdamOptimizer(learning_rate)
if batch_norm_momentum != 0:
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        training_op = optimizer.minimize(loss)
else:
    training_op = optimizer.minimize(loss, name='training_op')

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

init = tf.global_variables_initializer()
saver = tf.train.Saver()

X_train1 = X_train[y_train < 5]
y_train1 = y_train[y_train < 5]
X_valid1 = X_valid[y_valid < 5]
y_valid1 = y_valid[y_valid < 5]
X_test1 = X_test[y_test < 5]
y_test1 = y_test[y_test < 5]

n_epochs = 1000
batch_size = 10

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

with tf.Session() as sess:
    init.run()

    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train1))
        for rnd_indices in np.array_split(rnd_idx, len(X_train1) // batch_size):
            X_batch, y_batch = X_train1[rnd_indices], y_train1[rnd_indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training:True})
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid1, y: y_valid1})
        if loss_val < best_loss:
            save_path = saver.save(sess, 'Graph_Check_Points/my_mnist_model_0_to_4.ckpt')
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print('Early stopping!')
                break
        print('{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%'.format(
            epoch, loss_val, best_loss, acc_val * 100))

with tf.Session() as sess:
    saver.restore(sess, 'Graph_Check_Points/my_mnist_model_0_to_4.ckpt')
    acc_test = accuracy.eval(feed_dict={X: X_test1, y: y_test1})
    print('Final test accuracy: {:.2f}%'.format(acc_test * 100))

#%% Exercise 9
reset_graph()

n_inputs = 28 * 28
n_outputs = 5
batch_norm_momentum = 0.98
n_neurons = 160
'''
restore_saver = tf.train.import_meta_graph('Graph_Check_Points/my_mnist_model_0_to_4.ckpt.meta')
X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")
loss = tf.get_default_graph().get_tensor_by_name('loss:0')
accuracy = tf.get_default_graph().get_tensor_by_name('accuracy:0')

output_layer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='logits')
optimizer = tf.train.AdamOptimizer(learning_rate, name='Adam2')
training_op = optimizer.minimize(loss, var_list=output_layer_vars)
'''

he_init = tf.variance_scaling_initializer()
training = tf.placeholder_with_default(False, shape=(), name='training')

def dnn(inputs, n_hidden_layers=5, n_neurons=100, name=None,
        activation=tf.nn.elu, initializer=he_init,
        batch_norm_momentum = 0.999, training=training):
    with tf.variable_scope(name, default_name='dnn'):
        for layer in range(n_hidden_layers):
            inputs = tf.layers.dense(inputs, n_neurons, 
                                     kernel_initializer=initializer,
                                     name='hidden%d' % (layer + 1))
            if batch_norm_momentum != 0:
                inputs = tf.layers.batch_normalization(inputs,
                                                       momentum=batch_norm_momentum,
                                                       training=training)
            inputs = activation(inputs, name="hidden%d_out" % (layer + 1))
        return inputs

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None), name='y')
dnn_outputs = dnn(X, n_neurons=n_neurons,
                  batch_norm_momentum=batch_norm_momentum,
                  activation=tf.nn.relu, training=training)

logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=he_init, name='logits')
Y_proba = tf.nn.softmax(logits, name='Y_proba')
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name='loss')
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

optimizer = tf.train.AdamOptimizer(learning_rate)
train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dnn/hidden[45]|logits')
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    training_op = optimizer.minimize(loss, var_list=train_vars)

reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dnn/hidden[123]')
restore_saver = tf.train.Saver(reuse_vars)

init = tf.global_variables_initializer()
five_frozen_saver = tf.train.Saver()

X_train2_full = X_train[y_train >= 5]
y_train2_full = y_train[y_train >= 5] - 5
X_valid2_full = X_valid[y_valid >= 5]
y_valid2_full = y_valid[y_valid >= 5] - 5
X_test2 = X_test[y_test >= 5]
y_test2 = y_test[y_test >= 5] - 5

def sample_n_instances_per_class(X_in, y_in, n=100):
    Xs, ys = [], []
    for label in np.unique(y_in):
        idx = (y_in == label)
        Xc = X_in[idx][:n]
        yc = y_in[idx][:n]
        Xs.append(Xc)
        ys.append(yc)
    return np.concatenate(Xs), np.concatenate(ys)

X_train2, y_train2 = sample_n_instances_per_class(X_train2_full, y_train2_full, n=100)
X_valid2, y_valid2 = sample_n_instances_per_class(X_valid2_full, y_valid2_full, n=30)

learning_rate = 0.01
n_epochs = 1000
batch_size = 20

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, 'Graph_Check_Points/my_mnist_model_0_to_4.ckpt')
    
    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train2))
        for rnd_indices in np.array_split(rnd_idx, len(X_train2) // batch_size):
            X_batch, y_batch = X_train2[rnd_indices], y_train2[rnd_indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
            #sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid2, y: y_valid2})
        if loss_val < best_loss:
            save_path = five_frozen_saver.save(sess, 'Graph_Check_Points/my_mnist_model_5_to_9_five_frozen')
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print('Early stopping!')
                break
        print('{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%'.format(
            epoch, loss_val, best_loss, acc_val * 100))

    with tf.Session() as sess:
        five_frozen_saver.restore(sess, 'Graph_Check_Points/my_mnist_model_5_to_9_five_frozen')
        acc_test = accuracy.eval(feed_dict={X: X_test2, y: y_test2})
        print('Final test accuracy: {:.2f}%'.format(acc_test * 100))