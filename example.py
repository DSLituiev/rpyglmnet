#!/usr/bin/env python
from __future__ import print_function
from rpyglmnet import glmnet

print("simulating data")
import numpy as np
N = 20
P = 30
causal_ind = 4
beta = np.zeros((P))
beta[causal_ind] = 1.0
X = np.random.randn(*(N,P))
noise = np.random.randn(N)
y = X.dot(beta)

print("Initialize the model")
print("Option 1: use native glmnet `nfolds`")
model = glmnet(l1_ratio=0.5, n_folds=10)

print("Option 2: use `sklearn` `cv` syntax")

n_folds =10
try:
    from sklearn.model_selection import KFold
    kf = KFold(n_folds)
    model = glmnet(l1_ratio=0.5, cv=kf.get_n_splits(y), keep=True)
except:
    from sklearn.cross_validation import KFold
    kf = KFold(len(y), n_folds)
    model = glmnet(l1_ratio=0.5, cv=kf, keep=True)

print("Fit in sklearn style")
model.fit(X, y)

print("Predict in sklearn style")
y_hat = model.predict(X)
print("penalty", model.alpha_)

print("Use `.cross_val_score()` method in order to apply cross-validation metrics other than MSE")
from sklearn import metrics
print(model.cross_val_score(metrics.r2_score))

print("plot native R graphs")
model.rplot()

print("plot other metrics (requires `matplotlib`)")
model.plot()
