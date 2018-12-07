import tensorflow as tf
import numpy as np

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

#%% Creating and running a graph
reset_graph()

x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x * x * y + y + 2

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    print(result)

#%%
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result = f.eval()
    print(result)

#%% Managing graphs
reset_graph()

x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()

#%%
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

x2.graph is graph

#%%
x2.graph is tf.get_default_graph()

#%%
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)
    print(z_val)

#%% Linear Regression
from sklearn.datasets import fetch_california_housing

# Using the Normal Equation
reset_graph()

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()
    print(theta_value)
    print(' ')

# Compare with pure NumPy
X = housing_data_plus_bias
y = housing.target.reshape(-1, 1)
theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print(theta_numpy)
print(' ')

# Compare with Scikit-Learn
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing.data, housing.target.reshape(-1, 1))

print(np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T])
print(' ')

#%% Using Batch Gradient Descent
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

# Manually computing the gradients
reset_graph()
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
gradients = 2 / m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print('Epoch', epoch, 'MSE =', mse.eval())
        sess.run(training_op)
        
    best_theta = theta.eval()

print(best_theta)
print(' ')

# Using autodiff
reset_graph()
n_epochs = 1000
learning_rate = 0.01
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
gradients = tf.gradients(mse, theta)[0]
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print('Epoch', epoch, 'MSE =', mse.eval())
        sess.run(training_op)
        
    best_theta = theta.eval()

print(best_theta)
print(' ')

# Using a GradientDescentOptimizer
reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print('Epoch', epoch, 'MSE =', mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
    
print(best_theta)
print(' ')

#%% Feeding data to the training algorithm
# Placeholder nodes
reset_graph()

A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})

print(B_val_1)
print(B_val_2)
print(' ')

# Mini-batch Gradient Descent
n_epochs = 1000
learning_rate = 0.01

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n + 1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict = {X: X_batch, y: y_batch})
            
    best_theta = theta.eval()
    print(best_theta)
    print(' ')
    
#%% Saving and restoring a model
reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print('Epoch', epoch, 'MSE =', mse.eval())
            save_path = saver.save(sess, '/Users/100505/Documents/Python Scripts/graph/my_model_final.ckpt')
        sess.run(training_op)
    best_theta = theta.eval()
    save_path = saver.save(sess, '/Users/100505/Documents/Python Scripts/graph/my_model_final.ckpt')
    
#%%
with tf.Session() as sess:
    saver.restore(sess, '/Users/100505/Documents/Python Scripts/graph/my_model_final.ckpt')
    best_theta_restored = theta.eval()

#%%
reset_graph()
# notice that we start with an empty graph.

saver = tf.train.import_meta_graph('/Users/100505/Documents/Python Scripts/graph/my_model_final.ckpt.meta')  # this loads the graph structure
theta = tf.get_default_graph().get_tensor_by_name("theta:0") # not shown in the book

with tf.Session() as sess:
    saver.restore(sess, '/Users/100505/Documents/Python Scripts/graph/my_model_final.ckpt')  # this restores the graph's state
    best_theta_restored = theta.eval() # not shown in the book

#%% Visualizing the graph
reset_graph()

from datetime import datetime

now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = '{}/run-{}/'.format(root_logdir, now)

n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

batch_size = 100
n_batches = int(np.ceil(m / batch_size))
with tf.Session() as sess:                                                        # not shown in the book
    sess.run(init)                                                                # not shown

    for epoch in range(n_epochs):                                                 # not shown
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

file_writer.close()

#%% Name scopes
reset_graph()

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_epochs = 1000
learning_rate = 0.01
X = tf.placeholder(tf.float32, shape=(None, n + 1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
with tf.name_scope('loss') as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

file_writer.flush()
file_writer.close()

#%% Sharing Variables
reset_graph()
def relu(X):
    with tf.name_scope('relu'):
        w_shape = (int(X.get_shape()[1]), 1)                          # not shown in the book
        w = tf.Variable(tf.random_normal(w_shape), name='weights')    # not shown
        b = tf.Variable(0.0, name='bias')                             # not shown
        z = tf.add(tf.matmul(X, w), b, name='z')                      # not shown
        return tf.maximum(z, 0., name='max')                          # not shown
    
n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name='output')

file_writer = tf.summary.FileWriter('logs/relu2', tf.get_default_graph())
file_writer.close()

#%% Sharing Variables
reset_graph()

def relu(X):
    threshold = tf.get_variable('threshold', shape=(),
                                initializer=tf.constant_initializer(0.0))
    w_shape = (int(X.get_shape()[1]), 1)                        # not shown in the book
    w = tf.Variable(tf.random_normal(w_shape), name='weights')  # not shown
    b = tf.Variable(0.0, name='bias')                           # not shown
    z = tf.add(tf.matmul(X, w), b, name='z')                    # not shown
    return tf.maximum(z, threshold, name='max')

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
relus = []
for relu_index in range(5):
    with tf.variable_scope('relu', reuse=(relu_index >= 1)) as scope:
            relus.append(relu(X))
output = tf.add_n(relus, name='output')
file_writer = tf.summary.FileWriter('logs/relu9', tf.get_default_graph())
file_writer.close()

#%% Exercise
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime

m = 1000
X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)

plt.figure()
plt.plot(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1], 'go', label="Positive")
plt.plot(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1], 'r^', label="Negative")
plt.legend()

m, n = X_moons.shape
poly = PolynomialFeatures(3)
X_moon_expand = poly.fit_transform(X_moons)
X_moons_with_bias = np.c_[np.ones((m, 1)), X_moon_expand]
y_moons_column_vector = y_moons.reshape(-1, 1)

test_ratio = 0.2
test_size = int(m * test_ratio)
X_train = X_moons_with_bias[:-test_size]
X_test = X_moons_with_bias[-test_size:]
y_train = y_moons_column_vector[:-test_size]
y_test = y_moons_column_vector[-test_size:]

def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

reset_graph()
learning_rate = 0.01
logdir = log_dir("logreg")
n_features = X_moons_with_bias.shape[1]
X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
with tf.name_scope("logistic_regression"):
    with tf.name_scope("model"):
        theta = tf.Variable(tf.random_uniform([n_features, 1], -1.0, 1.0, seed=42), name='theta')
        y_proba = tf.math.sigmoid(tf.matmul(X, theta), name='predictions')
    with tf.name_scope("train"):
        loss = tf.losses.log_loss(y, y_proba)  # uses epsilon = 1e-7 by default
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)
        loss_summary = tf.summary.scalar('log_loss', loss)
    with tf.name_scope("init"):
        init = tf.global_variables_initializer()
    with tf.name_scope("save"):
        saver = tf.train.Saver()

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        
n_epochs = 10001
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = random_batch(X_train, y_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            
        loss_val, summary_str = sess.run([loss, loss_summary], feed_dict={X: X_test, y: y_test})
        file_writer.add_summary(summary_str, epoch)
        if epoch % 500 == 0:
            print("Epoch:", epoch, "\tLoss:", loss_val) 
    y_proba_val = y_proba.eval(feed_dict={X: X_test, y: y_test})
    file_writer.close()