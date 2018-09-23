#%% Import some useful packages
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# to make this notebook's output stable across runs
np.random.seed(42)

#%% Import MNIST
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')
x, y = mnist['data'], mnist['target']

#%%
some_digit = x[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation = 'nearest')
plt.axis('off')

#%%
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary, interpolation = 'nearest')
    plt.axis('off')

plot_digit(some_digit)

#%%
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

#%% Binary classifier
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(max_iter = 5, random_state = 42)
sgd_clf.fit(x_train, y_train_5)
sgd_clf.predict([some_digit])

#%%
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, x_train, y_train_5, cv = 3, scoring = 'accuracy')

#%%
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y = None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype = bool)

never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, x_train, y_train_5, cv = 3, scoring = 'accuracy')

#%%
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_5, cv = 3)
confusion_matrix(y_train_5, y_train_pred)

#%%
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)

#%%
recall_score(y_train_5, y_train_pred)

#%%
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)

#%%
y_scores = sgd_clf.decision_function([some_digit])
y_scores = cross_val_predict(sgd_clf, x_train, y_train_5, cv = 3, 
                             method = 'decision_function')

#%%
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

#%%
def plot_precision_recalll_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[: -1], 'b--', label = 'Precision', linewidth = 2)
    plt.plot(thresholds, recalls[: -1], 'g-', label = 'Recall', linewidth = 2)
    plt.xlabel('Threshold', fontsize = 16)
    plt.legend(loc = 'upper left', fontsize = 16)
    plt.ylim([0, 1])

plt.figure(figsize = (8, 4))
plot_precision_recalll_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-700000, 700000])

#%%
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, 'b', linewidth = 2)
    plt.xlabel('Recall', fontsize = 16)
    plt.ylabel('Precision', fontsize = 16)
    plt.axis([0, 1, 0, 1])

plt.figure(figsize = (8, 6))
plot_precision_vs_recall(precisions, recalls)

#%% ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label = None):
    plt.plot(fpr, tpr, linewidth = 2, label = label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize = 16)
    plt.ylabel('True Positive Rate', fontsize = 16)

plt.figure(figsize = (8, 6))
plot_roc_curve(fpr, tpr)

#%%
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)

#%%
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state = 42)
y_probas_forest = cross_val_predict(forest_clf, x_train, y_train_5, cv = 3, method = 'predict_proba')
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, threshold_forest = roc_curve(y_train_5, y_scores_forest)

#%%
plt.figure(figsize = (8, 6))
plt.plot(fpr, tpr, 'b:', linewidth = 2, label = 'SGD')
plot_roc_curve(fpr_forest, tpr_forest, 'Random Forest')
plt.legend(loc = 'lower right', fontsize = 16)

#%%
roc_auc_score(y_train_5, y_scores_forest)

#%%
y_train_pred_forest = cross_val_predict(forest_clf, x_train, y_train_5, cv = 3)
precision_score(y_train_5, y_train_pred_forest)

#%%
recall_score(y_train_5, y_train_pred_forest)

#%% Multiclass classification
sgd_clf.fit(x_train, y_train)
sgd_clf.predict([some_digit])
np.argmax(sgd_clf.decision_function([some_digit]))

#%%
cross_val_score(sgd_clf, x_train, y_train, cv = 3, scoring = 'accuracy')

#%%
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
x_train_scaled = scalar.fit_transform(x_train.astype(np.float64))
cross_val_score(sgd_clf, x_train_scaled, y_train, cv = 3, scoring = 'accuracy')

#%%
y_train_pred = cross_val_predict(sgd_clf, x_train_scaled, y_train, cv = 3)
conf_mx = confusion_matrix(y_train, y_train_pred)

#%%
plt.matshow(conf_mx, cmap=plt.cm.gray)

#%%
row_sums = conf_mx.sum(axis = 1, keepdims = True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)

#%% Exercise 1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
param_grid = [{'n_neighbors': [4], 'weights': ['distance']}]
knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv = 5, verbose = 3, n_jobs = -1)
grid_search.fit(x_train, y_train)

#%%
grid_search.best_score_

#%%
from sklearn.metrics import accuracy_score
y_predict = grid_search.predict(x_test)
accuracy_score(y_test, y_predict)

#%% Exercise 2
from scipy.ndimage.interpolation import shift
def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval = 0, mode = "constant")
    return shifted_image.reshape([-1])

#%%
x_train_augmented = [image for image in x_train]
y_train_augmented = [label for label in y_train]

#%%
for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, label in zip(x_train, y_train):
        x_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

x_train_augmented = np.array(x_train_augmented)
y_train_augmented = np.array(y_train_augmented)

#%%
knn_clf = KNeighborsClassifier(**grid_search.best_params_)
knn_clf.fit(x_train_augmented, y_train_augmented)
y_pred = knn_clf.predict(x_test)
accuracy_score(y_test, y_pred)