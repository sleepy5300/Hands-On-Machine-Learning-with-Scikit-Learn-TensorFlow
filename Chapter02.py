#%% Import some useful package
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from  sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from  sklearn.preprocessing import Imputer
from  future_encoders  import OrdinalEncoder
from future_encoders import OneHotEncoder
from future_encoders import ColumnTransformer
from  sklearn.pipeline  import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# To make this notebook's output stable across runs
np.random.seed(42)

#%% Get the data
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = os.path.join('datasets', 'housing')
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()

fetch_housing_data()

def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

housing = load_housing_data(HOUSING_PATH)

#%%
'''
housing.head()
housing.info()
housing['ocean_proximity'].value_counts()
housing.describe()
housing.hist(bins = 50, figsize = (15, 15))
'''
#%% To make sure this script's output identical at every run
np.random.seed(43)

#%%
'''
housing_with_id = housing.reset_index()
housing_with_id.head(10)
'''
#%% 
# Divide by 1.5 to limit the number of income categories
housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
# Label those above 5 as 5
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace = True)

#%%
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.iloc[train_index]
    strat_test_set = housing.iloc[test_index]

#%%
strat_train_set['income_cat'].value_counts() / len(strat_train_set)

#%%
strat_test_set['income_cat'].value_counts() / len(strat_test_set)

#%%
strat_train_set = strat_train_set.drop('income_cat', axis = 1)
strat_test_set = strat_test_set.drop('income_cat', axis = 1)

#%% Discover and visualize the data to gain insight
'''
housing = strat_train_set.copy()
housing.plot(kind = 'scatter', x = 'longitude', y = 'latitude', alpha = 0.1)
'''
#%%
'''
housing.plot(kind = 'scatter', x = 'longitude', y = 'latitude', alpha = 0.1,
             s = housing['population'] / 100, label = 'population', figsize = (10, 7),
             c = 'median_house_value', colormap = plt.get_cmap('jet'), colorbar = True, sharex = False)
'''
#%%
'''
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending = False)
'''
#%%
'''
attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize = (12, 8))
'''
#%%
'''
housing.plot(kind = 'scatter', x = 'median_income', y = 'median_house_value', alpha = 0.1)
plt.axis([0, 16, 0, 550000])
'''
#%%
'''
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household']=housing['population'] / housing['households']
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending = False)
'''
#%% Prepare the data for Machine Learning algorithms
housing = strat_train_set.drop('median_house_value', axis = 1)
housing_labels = strat_train_set['median_house_value']

#%%
'''
sample_incomplete_idx = housing.isnull().any(axis = 1)
sample_incomplete_rows = housing[sample_incomplete_idx]

# Option 1
sample_incomplete_rows.dropna(subset = ['total_bedrooms'])

# Option 2
sample_incomplete_rows.drop('total_bedrooms', axis = 1)

# Option 3
median = housing['total_bedrooms'].median()
sample_incomplete_rows['total_bedrooms'].fillna(median, inplace = True)
'''
#%%
'''
imputer = Imputer(strategy = 'median')
housing_num = housing.drop('ocean_proximity', axis = 1)
imputer.fit(housing_num)
imputer.statistics_

housing_num.median()
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns = housing_num.columns, index = list(housing.index.values))
housing_tr.loc[sample_incomplete_rows.index.values]
'''
#%%
'''
housing_cat = housing[['ocean_proximity']]
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

ordinal_encoder.categories_
cat_encoder = OneHotEncoder(sparse = True)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
'''
#%%
from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y = None):
        return self  # nothing else to do
    def transform(self, X, y = None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attribs = attr_adder.transform(housing.values)

housing_extra_attribshousing_ = pd.DataFrame(housing_extra_attribs,
    columns = list(housing.columns) + ['rooms_per_household', 'population_per_household'])

num_pipeline = Pipeline([('imputer', Imputer(strategy = 'median')),
                         ('attribs_adder', CombinedAttributesAdder()),
                         ('std_scaler', StandardScaler())])

#housing_num_tr = num_pipeline.fit_transform(housing_num)

#%%
'''
housing_tr['rooms_per_household'] = housing_tr['total_rooms'] / housing_tr['households']
housing_tr['population_per_household'] = housing_tr['population'] / housing_tr['households']
housing_tr['bedrooms_per_room'] = housing_tr['total_bedrooms'] / housing_tr['total_rooms']
num_pipeline = Pipeline([('imputer', Imputer(strategy = 'median')),
                         ('std_scalar', StandardScaler())])

num_pipeline = Pipeline([('std_scalar', StandardScaler())])
housing_num_tr = num_pipeline.fit_transform(housing_tr)
'''
#%%
'''
housing['total_bedrooms'] = housing_tr['total_bedrooms']
housing['rooms_per_household'] = housing_tr['rooms_per_household']
housing['population_per_household'] = housing_tr['population_per_household']
housing['bedrooms_per_room'] = housing_tr['bedrooms_per_room']
'''
num_attribs = list(housing)
num_attribs.remove('ocean_proximity')
cat_attribs = ['ocean_proximity']
full_pipeline = ColumnTransformer([('num', num_pipeline, num_attribs), 
                                   ('cat', OneHotEncoder(), cat_attribs)])
housing_prepared = full_pipeline.fit_transform(housing)

#%% Select and train a model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#%%
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)

#%%
lin_mae = mean_absolute_error(housing_labels, housing_predictions)

#%%
tree_reg = DecisionTreeRegressor(random_state = 42)
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)

#%% Fine-tune your model
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, 
                         scoring = 'neg_mean_squared_error', cv = 10)
tree_rmse_scores = np.sqrt(-scores)
scores = cross_val_score(lin_reg, housing_prepared, housing_labels, 
                         scoring = 'neg_mean_squared_error', cv = 10)
lin_rmse_scores = np.sqrt(-scores)

#%%
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(random_state = 42)
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)

#%%
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}]
forest_reg = RandomForestRegressor(random_state = 42)
grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, 
                           scoring = 'neg_mean_squared_error',
                           return_train_score = True)
grid_search.fit(housing_prepared, housing_labels)

#%%
grid_search.best_params_

#%%
grid_search.best_estimator_

#%%
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)

#%%
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distribs = {'n_estimators': randint(low = 1, high = 200),
                  'max_features': randint(low = 1, high = 8)}
forest_reg = RandomForestRegressor(random_state = 42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions = param_distribs, 
                                n_iter = 10, cv = 5, scoring = 'neg_mean_squared_error', 
                                random_state = 42)
rnd_search.fit(housing_prepared, housing_labels)

#%%
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)

#%%
feature_importances = grid_search.best_estimator_.feature_importances_

#%%
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop('median_house_value', axis = 1)
y_test = strat_test_set['median_house_value'].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

#%% Exercise 1
from sklearn.svm import SVR

svm_reg = SVR()
param_grid = [
    {'kernel': ['linear'], 'C' : [10.0, 1000., 1000.0, 10000.0]},
    {'kernel': ['rbf'], 'C': [1.0, 10.0, 100.0, 1000.0], 
     'gamma' : [0.01, 0.1, 1.0]}]

svm_search = GridSearchCV(svm_reg, param_grid, cv = 5,
                           scoring = 'neg_mean_squared_error',
                           verbose = 2, n_jobs = 4)
svm_search.fit(housing_prepared, housing_labels)
negative_mse = svm_search.best_score_
svm_rmse = np.sqrt(-negative_mse)

#%% Exercise 2
from scipy.stats import expon, reciprocal
param_distribs = {'kernel': ['linear', 'rbf'], 
                  'C': reciprocal(20, 200000), 
                  'gamma': expon(scale=1.0)}
svm_reg = SVR()
rnd_search = RandomizedSearchCV(svm_reg, param_distributions = param_distribs,
                                n_iter = 50, cv = 5, scoring = 'neg_mean_squared_error',
                                verbose = 2, n_jobs = 4, random_state = 42)
rnd_search.fit(housing_prepared, housing_labels)

#%% Exersice 3
from sklearn.base import BaseEstimator, TransformerMixin

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeasureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y = None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]

k = 5
top_k_feature_indices = indices_of_top_k(feature_importances, k)
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
np.array(attributes)[top_k_feature_indices]
