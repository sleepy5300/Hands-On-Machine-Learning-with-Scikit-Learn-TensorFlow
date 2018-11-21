#%% Training and visualizing
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:, 2 : ]  # petal length and width
Y = iris.target

tree_clf = DecisionTreeClassifier(max_depth = 2, random_state = 42)
tree_clf.fit(X, Y)

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, Y, axes = [0, 7.5, 0, 3], iris = True, legend = False, plot_training = True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    Y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, Y_pred, alpha = 0.3, cmap = custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, Y_pred, cmap = custom_cmap2, alpha = 0.8)
    if plot_training:
        plt.plot(X[:, 0][Y == 0], X[:, 1][Y == 0], "yo", label="Iris-Setosa")
        plt.plot(X[:, 0][Y == 1], X[:, 1][Y == 1], "bs", label="Iris-Versicolor")
        plt.plot(X[:, 0][Y == 2], X[:, 1][Y == 2], "g^", label="Iris-Virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize = 14)
        plt.ylabel("Petal width", fontsize = 14)
    else:
        plt.xlabel(r"$x_1$", fontsize = 18)
        plt.ylabel(r"$x_2$", fontsize = 18, rotation = 0)
    if legend:
        plt.legend(loc = "lower right", fontsize = 14)

plt.figure(figsize=(8, 4))
plot_decision_boundary(tree_clf, X, Y)
plt.plot([2.45, 2.45], [0, 3], "k-", linewidth = 2)
plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth = 2)
plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth = 2)
plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth = 2)
plt.text(1.20, 1.0, "Depth=0", fontsize = 15)
plt.text(3.2, 1.80, "Depth=1", fontsize = 13)
plt.text(3.8, 0.5, "(Depth=2)", fontsize = 11)

#%% Predicting class probabilities
tree_clf.predict_proba([[5, 1.5]])

#%% Predicting classes
tree_clf.predict([[5, 1.5]])

#%% Sensitivity to training set details
X[(X[:, 1]==X[:, 1][Y == 1].max()) & (Y == 1)] # widest Iris-Versicolor flower

#%%
not_widest_versicolor = (X[:, 1] != 1.8) | (Y == 2)
X_tweaked = X[not_widest_versicolor]
Y_tweaked = Y[not_widest_versicolor]

tree_clf_tweaked = DecisionTreeClassifier(max_depth = 2, random_state = 40)
tree_clf_tweaked.fit(X_tweaked, Y_tweaked)

plt.figure(figsize = (8, 4))
plot_decision_boundary(tree_clf_tweaked, X_tweaked, Y_tweaked, legend = False)
plt.plot([0, 7.5], [0.8, 0.8], "k-", linewidth = 2)
plt.plot([0, 7.5], [1.75, 1.75], "k--", linewidth = 2)
plt.text(1.0, 0.9, "Depth=0", fontsize = 15)
plt.text(1.0, 1.80, "Depth=1", fontsize = 13)

#%%
from sklearn.datasets import make_moons

Xm, Ym = make_moons(n_samples = 100, noise = 0.25, random_state = 53)

deep_tree_clf1 = DecisionTreeClassifier(random_state = 42)
deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf = 4, random_state = 42)
deep_tree_clf1.fit(Xm, Ym)
deep_tree_clf2.fit(Xm, Ym)

plt.figure(figsize = (11, 4))
plt.subplot(121)
plot_decision_boundary(deep_tree_clf1, Xm, Ym, axes = [-1.5, 2.5, -1, 1.5], iris = False)
plt.title('No restrictions', fontsize=16)
plt.subplot(122)
plot_decision_boundary(deep_tree_clf2, Xm, Ym, axes = [-1.5, 2.5, -1, 1.5], iris = False)
plt.title('min_samples_leaf = {}'.format(deep_tree_clf2.min_samples_leaf), fontsize = 14)

#%% Regression trees
# Quadratic training set + noise
np.random.seed(42)
m = 200
X = np.random.rand(m, 1)
Y = 4 * (X - 0.5) ** 2
Y = Y + np.random.randn(m, 1) / 10
plt.figure()
plt.plot(X, Y, 'b.')

#%%
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth = 2, random_state = 42)
tree_reg.fit(X, Y)

#%%
tree_reg1 = DecisionTreeRegressor(random_state = 42, max_depth = 2)
tree_reg2 = DecisionTreeRegressor(random_state = 42, max_depth = 3)
tree_reg1.fit(X, Y)
tree_reg2.fit(X, Y)

def plot_regression_predictions(tree_reg, X, Y, axes = [0, 1, -0.2, 1], ylabel = "$y$"):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    Y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$", fontsize = 18)
    if ylabel:
        plt.ylabel(ylabel, fontsize = 18, rotation = 0)
    plt.plot(X, Y, 'b.')
    plt.plot(x1, Y_pred, 'r.-', linewidth = 2, label = r"$\hat{y}$")

plt.figure(figsize = (11, 4))
plt.subplot(121)
plot_regression_predictions(tree_reg1, X, Y)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
plt.text(0.21, 0.65, "Depth=0", fontsize = 15)
plt.text(0.01, 0.2, "Depth=1", fontsize = 13)
plt.text(0.65, 0.8, "Depth=1", fontsize = 13)
plt.legend(loc="upper center", fontsize = 18)
plt.title("max_depth=2", fontsize = 14)

plt.subplot(122)
plot_regression_predictions(tree_reg2, X, Y, ylabel = None)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
    plt.plot([split, split], [-0.2, 1], style, linewidth = 2)
for split in (0.0458, 0.1298, 0.2873, 0.9040):
    plt.plot([split, split], [-0.2, 1], "k:", linewidth = 1)
plt.text(0.3, 0.5, "Depth=2", fontsize = 13)
plt.title("max_depth=3", fontsize = 14)

#%%
tree_reg1 = DecisionTreeRegressor(random_state = 42)
tree_reg2 = DecisionTreeRegressor(random_state = 42, min_samples_leaf = 10)
tree_reg1.fit(X, Y)
tree_reg2.fit(X, Y)

x1 = np.linspace(0, 1, 500).reshape(-1, 1)
Y_pred1 = tree_reg1.predict(x1)
Y_pred2 = tree_reg2.predict(x1)

plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.plot(X, Y, "b.")
plt.plot(x1, Y_pred1, "r.-", linewidth = 2, label = r"$\hat{y}$")
plt.axis([0, 1, -0.2, 1.1])
plt.xlabel("$x_1$", fontsize = 18)
plt.ylabel("$y$", fontsize = 18, rotation = 0)
plt.legend(loc="upper center", fontsize = 18)
plt.title("No restrictions", fontsize = 14)

plt.subplot(122)
plt.plot(X, Y, "b.")
plt.plot(x1, Y_pred2, "r.-", linewidth = 2, label = r"$\hat{y}$")
plt.axis([0, 1, -0.2, 1.1])
plt.xlabel("$x_1$", fontsize = 18)
plt.title("min_samples_leaf={}".format(tree_reg2.min_samples_leaf), fontsize = 14)