#%% Import some useful packages
import numpy as np
import matplotlib.pyplot as plt

# to make this notebook's output stable across runs
np.random.seed(42)

#%% Linear regression using the Normal Equation
X = 2 * np.random.rand(100, 1)
Y = 4 + 3 * X + np.random.rand(100, 1)
plt.plot(X, Y, 'b.')
plt.axis([0, 2, 0, 15])

#%%
X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.dot(np.dot(np.linalg.inv(np.dot(X_b.T, X_b)), X_b.T), Y)

#%%
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
Y_new = np.dot(X_new_b, theta_best)
plt.plot(X_new, Y_new, 'r-', label = 'Predictions')
plt.legend(loc = 'upper left', fontsize = 14)
plt.plot(X, Y, 'b.')

#%%
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
lin_reg.intercept_, lin_reg.coef_

#%% Linear regression using batch gradient descent
eta = 0.1
n_iterations = 1000
m = 100
theta = np.random.randn(2, 1)

for iteration in range(n_iterations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - Y)
    theta = theta - eta * gradients

theta_path_bgd = []

def plot_gradient_descent(theta, eta, theta_path = None):
    m = len(X_b)
    plt.plot(X, Y, 'b.')
    n_iterations = 1000
    for iteration in range(n_iterations):
        y_predict = X_new_b.dot(theta)
        style = 'r--'
        if iteration > 0:
            style = 'b-'
        plt.plot(X_new, y_predict, style)
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - Y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path_bgd.append(theta)
        
#%%
np.random.seed(42)
theta = np.random.randn(2, 1)  # random initialization
plot_gradient_descent(theta, eta = 0.1, theta_path = theta_path_bgd)

#%% Stochastic Gradient Descent
theta_path_sgd = []
m = len(X_b)
n_epochs  = 50
t0, t1 = 5, 50
theta = np.random.randn(2, 1)

def learning_schedule(t):
    return t0 / (t + t1)

for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i < 20:
            y_predict = X_new_b.dot(theta)
            if i > 0:
                style = 'g-'
            else:
                style = 'r--'
            plt.plot(X_new, y_predict, style)
        random_index = np.random.randint(m)
        xi = X_b[random_index : random_index + 1]
        yi = Y[random_index : random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)
plt.plot(X, Y, 'b.')
plt.axis([0, 2, 0, 15])

from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X, Y.ravel())
sgd_reg.intercept_, sgd_reg.coef_

#%% Mini-batch gradient descent
theta_path_mgd = []

n_iteration = 50
minibatch_size = 20
theta = np.random.randn(2, 1)

t0, t1 = 200, 1000
t = 0
for epoch in range(n_iteration):
    shuffled_indecies = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indecies]
    Y_shuffled = Y[shuffled_indecies]
    for i in range(0, m, minibatch_size):
        t += 1;
        xi = X_b_shuffled[i : i + minibatch_size]
        yi = Y_shuffled[i : i + minibatch_size]
        gradients = 2 / minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)

#%%
theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)

#%%
plt.figure(figsize=(7,4))
plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="Stochastic")
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="Mini-batch")
plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b-o", linewidth=3, label="Batch")
plt.legend(loc="upper left", fontsize=16)
plt.xlabel(r"$\theta_0$", fontsize=20)
plt.ylabel(r"$\theta_1$   ", fontsize=20, rotation=0)
plt.axis([2.5, 4.5, 2.3, 3.9])

#%% Polynomial regression
np.random.seed(42)

m = 100
X = 6 * np.random.rand(m, 1) - 3
Y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
plt.figure()
plt.plot(X, Y, 'b.')
plt.axis([-3, 3, 0, 10])

#%%
from sklearn.preprocessing import PolynomialFeatures

# Feature expansion
poly_features = PolynomialFeatures(degree = 2, include_bias = False)
X_poly = poly_features.fit_transform(X)

# Regressor
lin_reg = LinearRegression()
lin_reg.fit(X_poly, Y)
lin_reg.intercept_, lin_reg.coef_

#%%
plt.figure()
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.fit_transform(X_new)
Y_new = lin_reg.predict(X_new_poly)
plt.plot(X, Y, 'b.')
plt.plot(X_new, Y_new, 'r-', linewidth = 2, label = 'Predictions')
plt.legend(loc="upper left", fontsize=14)
plt.axis([-3, 3, 0, 10])

#%%
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

plt.figure()
for style, width, degree in(('g-', 1, 300), ('b--', 2, 2), ('r', 2, 1)):
    polybig_features = PolynomialFeatures(degree = degree, include_bias = False)
    std_scalar = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline([('poly_features', polybig_features), 
                                      ('std_scalar', std_scalar),
                                      ('lin_reg', lin_reg)])
    polynomial_regression.fit(X, Y)
    Y_newbig = polynomial_regression.predict(X_new)
    plt.plot(X_new, Y_newbig, style, label = str(degree), linewidth = width)

plt.plot(X, Y, 'b.')
plt.legend(loc = 'upper left')
plt.axis([-3, 3, 0, 10])

#%%
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth = 2, label = "train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth = 3, label = "val")
    plt.legend(loc = "upper right", fontsize = 14)   # not shown in the book
    plt.xlabel("Training set size", fontsize = 14) # not shown
    plt.ylabel("RMSE", fontsize = 14)              # not shown

lin_reg = LinearRegression()
plt.figure()
plot_learning_curves(lin_reg, X, Y)
plt.axis([0, 80, 0, 3])                         # not shown in the book

polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree = 10, include_bias = False)),
        ("lin_reg", LinearRegression())])
    
plt.figure()
plot_learning_curves(polynomial_regression, X, Y)
plt.axis([0, 80, 0, 3])           # not shown

#%% Regularized models
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
Y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)

def plot_model(model_class, polynomial, alphas, **model_kargs):
    for alpha, style in zip(alphas, ("b-", "g--", "r:")):
        model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline([('poly_features', PolynomialFeatures(degree = 10, include_bias = False)),
                              ('std_scalar', StandardScaler()),
                              ("regul_reg", model)])
        model.fit(X, Y)
        Y_new_regul = model.predict(X_new)
        lw = 2 if alpha > 0 else 1
        plt.plot(X_new, Y_new_regul, style, linewidth = lw, label = r"$\alpha = {}$".format(alpha))
    plt.plot(X, Y, "b.", linewidth = 3)
    plt.legend(loc = "upper left", fontsize = 15)
    plt.xlabel("$x_1$", fontsize = 18)
    plt.axis([0, 3, 0, 4])

plt.figure(figsize=(8, 4))
plt.subplot(121)
plot_model(Ridge, polynomial = False, alphas = (0, 10, 100), random_state = 42)
plt.ylabel("$y$", rotation = 0, fontsize = 18)
plt.subplot(122)
plot_model(Ridge, polynomial = True, alphas = (0, 10**-5, 1), random_state = 42)

#%%
np.random.seed(42)
m = 100;
X = 6 * np.random.rand(m, 1) - 3
Y = 2 + X + 0.5 * X**2 + np.random.randn(m, 1)

X_train, X_val, Y_train, Y_val = train_test_split(X[:50], Y[:50].ravel(), test_size = 0.5, random_state = 10)

poly_scalar = Pipeline([
        ('poly_features', PolynomialFeatures(degree = 90, include_bias = False)),
        ('std_scalar', StandardScaler())])

X_train_poly_scaled = poly_scalar.fit_transform(X_train)
X_val_poly_scaled = poly_scalar.transform(X_val)

sgd_reg = SGDRegressor(max_iter = 1,
                       penalty = None,
                       eta0 = 0.0005,
                       warm_start = True,
                       learning_rate = "constant",
                       random_state = 42)

n_epochs = 500
train_errors, val_errors = [], []
for epoch in range(n_epochs):
    sgd_reg.fit(X_train_poly_scaled, Y_train)
    Y_train_predict = sgd_reg.predict(X_train_poly_scaled)
    Y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    train_errors.append(mean_squared_error(Y_train, Y_train_predict))
    val_errors.append(mean_squared_error(Y_val, Y_val_predict))
    
best_epoch = np.argmin(val_errors)
best_val_rmse = np.sqrt(val_errors[best_epoch])

plt.figure()

plt.annotate('Best model',
             xy = (best_epoch, best_val_rmse), 
             xytext = (best_epoch, best_val_rmse + 1),
             ha = "center",
             arrowprops = dict(facecolor = 'black', shrink = 0.05),
             fontsize = 16)
best_val_rmse -= 0.03  # just to make the graph look better
plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth = 2)
plt.plot(np.sqrt(val_errors), "b-", linewidth = 3, label = "Validation set")
plt.plot(np.sqrt(train_errors), "r--", linewidth = 2, label = "Training set")
plt.legend(loc="upper right", fontsize = 14)
plt.xlabel("Epoch", fontsize = 14)
plt.ylabel("RMSE", fontsize = 14)

#%% Logistic regression
t = np.linspace(-10, 10, 100)
sig = 1 / (1 + np.exp(-t))
plt.figure()
plt.plot(t, sig)
plt.plot([-10, 10], [0, 0], "k-")
plt.plot([-10, 10], [0.5, 0.5], "k:")
plt.plot([-10, 10], [1, 1], "k:")
plt.plot([0, 0], [-1.1, 1.1], "k-")
plt.axis([-10, 10, -0.1, 1.1])

#%%
from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())

#%%
X = iris['data'][:, 3].reshape(-1, 1)       # petal width
Y = (iris['target'] == 2).astype(np.int)    # 1 if Iris-Virginica, else 0

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state = 42)
log_reg.fit(X, Y)

#%%
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
Y_proba = log_reg.predict_proba(X_new)

plt.figure()
plt.plot(X_new, Y_proba[:, 1], "g-", linewidth = 2, label = "Iris-Virginica")
plt.plot(X_new, Y_proba[:, 0], "b--", linewidth = 2, label = "Not Iris-Virginica")

decision_boundary = X_new[Y_proba[:, 1] >= 0.5][0]
plt.figure(figsize=(8, 3))
plt.plot(X[Y == 0], Y[Y == 0], "bs")
plt.plot(X[Y == 1], Y[Y == 1], "g^")
plt.plot([decision_boundary, decision_boundary], [-0.5, 1.5], "k:", linewidth = 2)
plt.plot(X_new, Y_proba[:, 1], "g-", linewidth = 2, label = "Iris-Virginica")
plt.plot(X_new, Y_proba[:, 0], "b--", linewidth = 2, label = "Not Iris-Virginica")
plt.text(decision_boundary + 0.02, 0.15, "Decision  boundary", fontsize = 14, color = "k", ha = "center")
plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width = 0.05, head_length = 0.1, fc = 'b', ec = 'b')
plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width = 0.05, head_length = 0.1, fc = 'g', ec = 'g')
plt.xlabel("Petal width (cm)", fontsize = 14)
plt.ylabel("Probability", fontsize = 14)
plt.legend(loc = "center left", fontsize = 14)
plt.axis([0, 3, -0.02, 1.02])

#%%
X = iris['data'][:, (2, 3)]     # petal length, petal width
Y = (iris['target'] == 2).astype(np.int)

log_reg = LogisticRegression(C = 10**10, random_state = 42)
log_reg.fit(X, Y)

x0, x1 = np.meshgrid(np.linspace(2.9, 7, 500).reshape(-1, 1),
                     np.linspace(0.8, 2.7, 200).reshape(-1, 1))

X_new = np.c_[x0.ravel(), x1.ravel()]
Y_proba = log_reg.predict_proba(X_new)

plt.figure(figsize=(10, 4))
plt.plot(X[Y == 0, 0], X[Y == 0, 1], 'bs')
plt.plot(X[Y == 1, 0], X[Y == 1, 1], 'g^')

zz = Y_proba[:, 1].reshape(x0.shape)
contour = plt.contour(x0, x1, zz, cmap = plt.cm.brg)

left_right = np.array([2.9, 7])
boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]

plt.clabel(contour, inline = 1, fontsize = 12)
plt.plot(left_right, boundary, "k--", linewidth = 3)
plt.text(3.5, 1.5, "Not Iris-Virginica", fontsize = 14, color = "b", ha = "center")
plt.text(6.5, 2.3, "Iris-Virginica", fontsize = 14, color = "g", ha = "center")
plt.xlabel("Petal length", fontsize = 14)
plt.ylabel("Petal width", fontsize = 14)
plt.axis([2.9, 7, 0.8, 2.7])

#%% Softmax classifier
X = iris['data'][:, (2, 3)]     # petal length, petal width
Y = (iris['target'])

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42)
softmax_reg.fit(X, Y)

x0, x1 = np.meshgrid(np.linspace(0, 8, 500).reshape(-1, 1),
                     np.linspace(0, 3.5, 200).reshape(-1, 1))
X_new = np.c_[x0.ravel(), x1.ravel()]


Y_proba = softmax_reg.predict_proba(X_new)
Y_predict = softmax_reg.predict(X_new)
zz1 = Y_proba[:, 1].reshape(x0.shape)
zz = Y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[Y == 2, 0], X[Y == 2, 1], "g^", label = "Iris-Virginica")
plt.plot(X[Y == 1, 0], X[Y == 1, 1], "bs", label = "Iris-Versicolor")
plt.plot(X[Y == 0, 0], X[Y == 0, 1], "yo", label = "Iris-Setosa")

from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

plt.contourf(x0, x1, zz, cmap = custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap = plt.cm.brg)
plt.clabel(contour, inline = 1, fontsize = 12)
plt.xlabel("Petal length", fontsize = 14)
plt.ylabel("Petal width", fontsize = 14)
plt.legend(loc = "center left", fontsize = 14)
plt.axis([0, 7, 0, 3.5])