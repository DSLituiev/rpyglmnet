## rpyglmnet
a `rpy2` wrapper around `R glmnet` package.

## depends on

- numpy
- [rpy2](http://rpy2.bitbucket.org/)
- [glmnet](https://github.com/jeffwong/glmnet)
- [sklearn](http://scikit-learn.org/stable/)

## Purpose

This wrapper makes popular `glmnet` `R` package available with all its parameters. 
It also makes it compatible with `sklearn` methods as a subclass of `sklearn.base.BaseEstimator` class,
which makes it possible to use it in `sklearn` pipelines.

## Nota bene

In this wrapper, the parameter names are chosen to disambiguate `R` and `sklearn.ElasticNet` naming standards:

- **The regularization penalty** can be fed as `lambda_` (and accessed as `alpha_` and `alphas_`);

- **The ratio of L1 penalty** to the total `L1 + L2` penalty mix is denoted as `l1_ratio` as in `sklearn` package;

- full stop `.`  in various `R` parameters is replaced by underscore `_`;

## Interfaces to scikit-learn

- use standard `.fit()` and `.predict()` methods:
- cross-validation can be fed as `cv` parameter as well as `R` style `foldid` list
- to get cross-validation metrics in cross-validation mode:
 + initialize with `keep=True` flag, e.g.: `model = glmnet(..., cv=5, keep=True)`
 + call `model.cross_val_score(mymetric)`, e.g.: `model.cross_val_score(sklearn.metrics.r2_score)` 

## Example

See `example.py`
