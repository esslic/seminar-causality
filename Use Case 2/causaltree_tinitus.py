# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 09:30:50 2020

@author: Christoph
"""
#%%
import os
import pandas as pd
import urllib.request
import numpy as np
from sklearn.preprocessing import StandardScaler
import econml
from econml.causal_forest import CausalForest
from econml.sklearn_extensions.linear_model import WeightedLassoCVWrapper, WeightedLasso, WeightedLassoCV
import matplotlib.pyplot as plt

data = pd.read_csv('tinnitus.csv', sep = ";")


#%%

# Defintion of effect variable
Y = data['question6'].values

# Defintion of treatment variable
T = data["question2"].values

# Prepare data
scaler = StandardScaler()
W1 = scaler.fit_transform(data[[c for c in data.columns if c not in ['question2', 'question6', 'tschq_10_perceive',
                                                                    'tschq_36_manifest']]].values)
W2 = pd.get_dummies(data[['tschq_10_perceive']]).values
W3 = pd.get_dummies(data[['tschq_36_manifest']]).values
W = np.concatenate([W1, W2, W3], axis=1)

X = data[['tfsum']].values


#%%
# Definition of causal tree parameters
n_trees = 1000
min_leaf_size = 50
max_depth = 20
subsample_ratio = 0.04

#%%
# Definition of range of variable tested for heterogeneity
min_tfsum = 0.0
max_tfsum = 24.0
delta = (max_tfsum - min_tfsum) / 100
X_test = np.arange(min_tfsum, max_tfsum + delta - 0.001, delta).reshape(-1, 1)


#%%
# Estimation of causal tree
est = CausalForest(n_trees=n_trees, min_leaf_size=min_leaf_size, max_depth=max_depth,
                    subsample_ratio=subsample_ratio,
                    model_T=WeightedLassoCVWrapper(cv=3),
                    model_Y=WeightedLassoCVWrapper(cv=3),
                    random_state=123)
est.fit(Y, T, X=X, W=W)
treatment_effects = est.effect(X_test)
te_lower, te_upper = est.effect_interval(X_test)

#%%
# Plot results
plt.figure(figsize=(15, 5))
plt.plot(X_test.flatten(), treatment_effects)
plt.fill_between(X_test.flatten(), te_lower, te_upper, label = "90% CI", alpha=0.3)
plt.xlabel('Level of psychological distress', fontsize = 15)
plt.ylabel('Level of stress due to tinnitus', fontsize = 15)
plt.legend()
plt.title("HTE in Stress Level due to Tinnitus", fontsize = 18)
plt.show()

