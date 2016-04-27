import sys
import logging
import warnings
import numpy as np
from sklearn import base, cross_validation, model_selection
try:
    import pandas as pd
except:
    pass

try:
    import matplotlib.pyplot as plt
except:
    warnings.warn("Could not import matplotlib. Plotting will not work")


from rpy2.robjects import FloatVector, IntVector
from rpy2.robjects import Matrix
from rpy2 import robjects
from rpy2.robjects import pandas2ri
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
pandas2ri.activate()

#####################################################################
# R package names
packnames = ('glmnet',)

# import rpy2's package module
import rpy2.robjects.packages as rpackages

if all(rpackages.isinstalled(x) for x in packnames):
    have_tutorial_packages = True
else:
    have_tutorial_packages = False
    if not have_tutorial_packages:
        # import R's utility package
        utils = rpackages.importr('utils')
        # select a mirror for R packages
        utils.chooseCRANmirror(ind=1) # select the first mirror in the list
#We are now ready to install packages using R's own function install.package:
if not have_tutorial_packages:
    # R vector of strings
    from rpy2.robjects.vectors import StrVector
    # file
    packnames_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    if len(packnames_to_install) > 0:
        utils.install_packages(StrVector(packnames_to_install))
#####################################################################
def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False
run_from_ipython()

def rplotwrap(func, dpi = 150, width=1024, height=896):
    """ ipython wrapper for plotting functions; use:
                @rplotwrap
                def plot(self):
                    rplot = robjects.r('plot')
                    rplot(self.rmodel)
                    return
    """

    def dorplot(*args, **kwargs):
        """Arguments:
            file -- output file [default: tmp.png]
            ...     --
        """
        if "file" in kwargs:
            file = kwargs.pop("file")
        else:
            file = "tmp.png"
        from rpy2.robjects.packages import importr
        grdevices = importr('grDevices')
        grdevices.png(file=file, width=width, height=height, res = dpi)
        result = func(*args, **kwargs)
        grdevices.dev_off()

        if run_from_ipython():
            from IPython.display import Image, display
            display(Image(filename=file))
            return result
    return dorplot
#################################################################
def is_basekfold(cv):
    return isinstance(cv, cross_validation._BaseKFold ) or isinstance(cv, model_selection._split._BaseKFold)

def assert_basekfold(cv):
    assert is_basekfold(cv)

def get_foldid(cv):
    assert_basekfold(cv)
    foldid = np.empty(cv.n, dtype=np.int32)
    for nn, (_, tsi) in enumerate(cv):
        foldid[tsi] = nn
    return foldid

#################################################################
rmatrix = robjects.r('as.matrix')

class ArgumentNotSetError(ValueError):
    pass

class glmnet(base.BaseEstimator):
    """Fit a generalized linear model via penalized maximum likelihood.
    The regularization path is computed for the lasso or elasticnet
    penalty at a grid of values for the regularization parameter lambda. 
    Can deal with all shapes of data, including very large 
    sparse data matrices. Fits linear, logistic and multinomial, 
    poisson, and Cox regression models.

    Usage
    -----

        glmnet(x, y, 
            family=["gaussian","binomial","poisson","multinomial","cox","mgaussian"],
            weights, offset=NULL, l1_ratio = 1, nlambda = 100,
            lambda_min_ratio = ifelse(nobs<nvars,0.01,0.0001), 
            lambda_=NULL,
            standardize = TRUE, intercept=TRUE, thresh = 1e-07,  dfmax = nvars + 1,
            pmax = min(dfmax * 2+20, nvars),
            exclude, penalty.factor = rep(1, nvars),
            lower_limits=-Inf, upper_limits=Inf, maxit=100000,
            type_gaussian=ifelse(nvars<500,"covariance","naive"),
            type_logistic=c("Newton","modified.Newton"),
            standardize_response=FALSE, 
            type_multinomial=c("ungrouped","grouped"))

    Arguments
    ----------

        x       
            input matrix, of dimension nobs x nvars; each row is an observation vector. 
            Can be in sparse matrix format (inherit from class "sparseMatrix" as in 
            package Matrix; not yet available for family="cox")
        y   
            response variable. Quantitative for family="gaussian", or family="poisson"
            (non-negative counts). For family="binomial" should be either a factor 
            with two levels, or a two-column matrix of counts or proportions (the second
            column is treated as the target class; for a factor, the last level in 
            alphabetical order is the target class). For family="multinomial",
            can be a nc>=2 level factor, or a matrix with nc columns of counts
            or proportions. For either "binomial" or "multinomial", if y is presented 
            as a vector, it will be coerced into a factor. For family="cox", y should 
            be a two-column matrix with columns named 'time' and 'status'. The latter is a
            binary variable, with '1' indicating death, and '0' indicating right censored. 
            The function Surv() in package survival produces such a matrix. 
            For family="mgaussian", y is a matrix of quantitative responses.

        family  
            Response type (see above)

        weights 
            observation weights. Can be total counts if responses are proportion matrices. 
            Default is 1 for each observation

        offset  
            A vector of length nobs that is included in the linear predictor 
            (a nobs x nc matrix for the "multinomial" family). 
            Useful for the "poisson" family (e.g. log of exposure time),
            or for refining a model by starting at a current fit. Default is NULL. 
            If supplied, then values must also be supplied to the predict function.

        l1_ratio    
            The elasticnet mixing parameter, with 0≤α≤ 1. The penalty is defined as
                    (1-α)/2||β||_2^2+α||β||_1.
            alpha=1 is the lasso penalty, and alpha=0 the ridge penalty.

        nlambda 
            The number of lambda values - default is 100.

        lambda.min.ratio    
            Smallest value for lambda, as a fraction of `lambda_max`, 
            the (data derived) entry value (i.e. the smallest value 
            for which all coefficients are zero). 
            The default depends on the sample size `nobs` 
            relative to the number of variables nvars. 
            If `nobs > nvars`, the default is 0.0001, close to zero.
            If `nobs < nvars`, the default is 0.01. 
            A very small value of lambda.min.ratio will lead 
            to a saturated fit in the nobs < nvars case. 
            This is undefined for "binomial" and "multinomial" models,
            and glmnet will exit gracefully when the percentage d
            eviance explained is almost 1.

        lambda_
            A user supplied lambda sequence. Typical usage is to have the program 
            compute its own lambda sequence based on nlambda and lambda.min.ratio. 
            Supplying a value of lambda overrides this. WARNING: use with care. 
            Do not supply a single value for lambda (for predictions after CV use 
            predict() instead). Supply instead a decreasing sequence of lambda values.
            glmnet relies on its warms starts for speed, and its often faster to fit 
            a whole path than compute a single fit.

        standardize 
            Logical flag for x variable standardization, prior to fitting 
            the model sequence. The coefficients are always returned 
            on the original scale. Default is standardize=TRUE. 
            If variables are in the same units already, 
            you might not wish to standardize. See details below for 
            y standardization with family="gaussian".

        intercept OR fit_intercept
            Should intercept(s) be fitted (default=True) or set to zero (FALSE)
        thresh  
            Convergence threshold for coordinate descent.
            Each inner coordinate-descent loop continues 
            until the maximum change in the objective
            after any coefficient update is less than 
            `thresh` times the null deviance. Defaults value is 1e-7.
        dfmax   
            Limit the maximum number of variables in the model. 
            Useful for very large nvars, if a partial path is desired.
        pmax    
            Limit the maximum number of variables ever to be nonzero
        exclude 
            Indices of variables to be excluded from the model. 
            Default is none. Equivalent to an infinite penalty factor (next item).
        penalty.factor  
            Separate penalty factors can be applied to each coefficient. 
            This is a number that multiplies lambda 
            to allow differential shrinkage. Can be 0 for some variables,
            which implies no shrinkage, and that variable is always 
            included in the model. Default is 1 for all variables 
            (and implicitly infinity for variables listed in exclude). 
            Note: the penalty factors are internally rescaled 
            to sum to `nvars`, and the lambda sequence will reflect this change.
        lower.limits    
            Vector of lower limits for each coefficient; default -Inf. 
            Each of these must be non-positive. Can be presented
            as a single value (which will then be replicated), 
            else a vector of length nvars
        upper.limits    
            Vector of upper limits for each coefficient; default Inf. See lower.limits
        maxit   
            Maximum number of passes over the data for all lambda values; default is 10^5.
        type_gaussian   
            Two algorithm types are supported 
            for (only) family="gaussian". The default when nvar<500 
            is type_gaussian="covariance", and saves all
            inner-products ever computed. This can be much faster 
            than type_gaussian="naive", which loops through `nobs`
            every time an inner-product is computed. The latter can be
            far more efficient for nvar >> nobs situations, or when nvar > 500.
        type_logistic   
            If "Newton" then the exact hessian is used (default),
            while "modified.Newton" uses an upper-bound 
            on the hessian, and can be faster.
        standardize.response    
            This is for the family="mgaussian" family, 
            and allows the user to standardize the response variables
        type.multinomial    
            If "grouped" then a grouped lasso penalty is used on the 
            multinomial coefficients for a variable. This ensures 
            they are all in our out together. The default is "ungrouped"

    Cross-validation arguments:
        nfolds  
            number of folds - default is 10. Although nfolds can be as large
            as the sample size (leave-one-out CV), it is not recommended for 
            large datasets. Smallest value allowable is nfolds=3
        foldid  
            an optional vector of values between 1 and nfold identifying 
            what fold each observation is in. If supplied, nfold can be missing
        type_measure    
            loss to use for cross-validation. Currently five options, 
            not all available for all models. The default is 
            `type_measure="deviance"`, which uses squared-error 
            for gaussian models (a.k.a type.measure="mse" there), 
            deviance for logistic and poisson regression,
            and partial-likelihood for the Cox model. 
            `type.measure="class"` applies to binomial and multinomial logistic
            regression only, and gives misclassification error.
            `type.measure="auc"` is for two-class logistic regression only,
            and gives area under the ROC curve. 
            `type.measure="mse"` or `type.measure="mae"` (mean absolute error)
            can be used by all models except the "cox"; 
            they measure the deviation from the fitted mean to the response.
        grouped
            This is an experimental argument, with default TRUE,
            and can be ignored by most users. For all models except the "cox",
            this refers to computing nfolds separate statistics, and then 
            using their mean and estimated standard error to describe the CV
            curve. If `grouped=FALSE`, an error matrix is built up 
            at the observation level from the predictions from the `nfold` fits,
            and then summarized (does not apply to `type_measure="auc"`).
            For the "cox" family, `grouped=TRUE` obtains the CV partial
            likelihood for the Kth fold by subtraction; by subtracting
            the log partial likelihood evaluated on the full dataset 
            from that evaluated on the on the `(K-1)/K` dataset. 
            This makes more efficient use of risk sets.
            With `grouped=FALSE` the log partial likelihood is computed 
            only on the Kth fold
        keep
            If keep=TRUE, a prevalidated array is returned containing fitted
            values for each observation and each value of lambda. 
            This means these fits are computed with this observation and 
            the rest of its fold omitted. The folid vector is also returned.
            Default is `keep=FALSE`
        parallel
            If TRUE, use parallel foreach to fit each fold. 
            Must register parallel before hand, such as doMC or others.
            See the example below.

        Attributes [cross-validation]
        -----------------------------

        lambda
            the values of lambda used in the fits.
        cvm
            The mean cross-validated error - a vector of length length(lambda).
        cvsd
            estimate of standard error of cvm.
        cvup
            upper curve = cvm+cvsd.
        cvlo
            lower curve = cvm-cvsd.
        nzero
            number of non-zero coefficients at each lambda.
        name
            a text string indicating type of measure (for plotting purposes).
        glmnet.fit
            a fitted glmnet object for the full data.
        lambda.min
            value of lambda that gives minimum cvm.
        lambda.1se
            largest value of lambda such that error is within 1 standard error of the minimum.
        fit.preval
            if keep=TRUE, this is the array of prevalidated fits. Some entries can be NA, if that and subsequent values of lambda are not reached for that fold
        foldid
            if keep=TRUE, the fold assignments used
    """

    def __init__(self,
                n_folds=0, nfolds=0, cv=0, foldid=None,
                which_coef="1se",
                family=None,#c("gaussian","binomial","poisson","multinomial","cox","mgaussian"),
                weights=None, offset=None, l1_ratio=1.0, nlambda = 100,
                lambda_min_ratio = None, lambda_=None,
                standardize = True, intercept=True, fit_intercept= True,
                thresh = 1e-07, # dfmax = nvars + 1,
                #pmax = min(dfmax * 2+20, nvars), 
                exclude=None,
                penalty_factor = None, #rep(1, nvars),
                lower_limits=-np.inf,
                upper_limits=np.inf, maxit=100000,
                #type_gaussian=ifelse(nvars<500,"covariance","naive"),
                #type_logistic=c("Newton","modified.Newton"),
                standardize_response=False,
                #type.multinomial=c("ungrouped","grouped"),
                keep = False,
                ):

        self.intercept = intercept and fit_intercept
        self.fit_intercept = self.intercept

        self.params = dict(
                nfolds= max(n_folds, nfolds, cv if type(cv) in (int,float) else 0) if lambda_ is None else 0,
                alpha=l1_ratio,
                nlambda=nlambda,
                intercept = self.intercept,
                thresh = thresh,
                maxit = maxit,
                standardize_response = standardize_response,
                standardize = standardize,
                keep = keep,
                            )
        self.keep = keep
        self.thresh=thresh
        self.maxit = maxit
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        self.penalty_factor = penalty_factor
        self.standardize = standardize
        self.standardize_response = standardize_response
        self.nlambda = nlambda
        self.l1_ratio = l1_ratio
        #self.cv = cv

        self.family = family
        if family is not None:
            self.params["family"] = family

        self.weights = weights
        if weights is not None:
            self.params["weights"] = weights

        self.offset = offset
        if offset is not None:
            self.params["offset"] = offset

        self.exclude = exclude
        if exclude is not None:
            self.params["exclude"] = exclude

        self.lambda_min_ratio = lambda_min_ratio
        if lambda_min_ratio is not None:
            self.params["lambda_min_ratio"] = lambda_min_ratio

        self.penalty_factor = penalty_factor
        if penalty_factor is not None:
            self.params["penalty_factor"] = penalty_factor

        self.params = dict( [ (kk.replace("_","."), vv) for kk, vv in self.params.items() ] )

        if is_basekfold(cv):
            self.cv = cv
            self._foldid = get_foldid(cv)
            foldid = self._foldid
            #self.params["nfolds"] = cv.n_folds
            #self.nfolds = cv.n_folds
        else:
            self.nfolds = self.params["nfolds"]
        #self.n_folds = self.params["nfolds"]
        #self.cv = self.params["nfolds"]

        self.foldid = np.array(foldid, dtype=int) if foldid is not None else None
        if self.foldid is not None:
            self.foldid = self.foldid - min(self.foldid)
            self.params["foldid"] = 1 + self.foldid
            self.nfolds = np.unique(self.foldid).shape[0]
            self.params["nfolds"] = self.nfolds

        self.params["alpha"] = l1_ratio

        self.lambda_ = lambda_
        if lambda_ is not None:
            self.params["lambda"] = lambda_

        if self.params["nfolds"] <= 1:
            self.cv = self.params.pop("nfolds")
        else:
            self.__dict__["n_folds"] = self.params["nfolds"]
            self.__dict__["nfolds"] = self.params["nfolds"]
        """ other options include "all" and  "best" """
        self.which_coef = which_coef
        
        if family in ("binomial", "multinomial", "cox"):
            self._estimator_type = 'classifier'
        else:
            self._estimator_type = 'regressor'

        if ("nfolds" in self.params) and (self.params["nfolds"] != 0):
            self.params["keep"] = True
            self.fun = "glmnet::cv.glmnet"
            logging.debug("running %u fold cross-validation" % self.params["nfolds"] )
        else:
            self.params.pop("keep")
            #self.params.pop("cv")
            if "nfolds" in self.params:
                self.params.pop("nfolds")
            self.fun = "glmnet::glmnet"
        #print(self.fun)
        self._glmnet_ = robjects.r(self.fun, )
        #logging.debug("running %s with parameters:\n%s" % ( self.fun, "\n".join(["%s\t%s" % (kk, repr(vv)) for kk, vv in self.params.items() ]) ))

    def __repr__(self):
        out = "glmnet[wrapped R function '%s'](\n" % self.fun 
        for kk, vv in self.params.items():
            out += "%s:\t%s\n" % (kk, vv)
        out += "\t)"
        return out

    def __setattr__(self, name, value):
        if name in  ("cv"):
            if type(value) in (int, float):
                for nn in ("n_folds", "nfolds",):
                    self.__dict__[nn] = value
            elif is_basekfold(value):
                for nn in ("n_folds", "nfolds",):
                    self.__dict__[nn] = value.n_folds
            
        if name in  ("n_folds", "nfolds",):
            logging.debug("setting folds")
            for nn in ("n_folds", "nfolds",):
                self.__dict__[nn] = value
            if "cv" in self.__dict__ and is_basekfold(self.cv):
                #raise ValueError("resetting cv to integer value\t%u" % value )
                warnings.warn("resetting cv to integer value\t%u" % value)
            self.__dict__["cv"] = value

        else:
            self.__dict__[name] = value

    #@property
    #def __doc__(self):
    #    rhelp = robjects.r("help")
    #    return str(rhelp(self.fun.split(":")[-1], package = "glmnet") )

    def fit(self, X, y,  l1_ratio = 1 , **fit_params):
        X = np.asarray(X)
        y = np.asarray(y)
        #logging.debug( repr(self.params) )
        self.y = y if len(y.shape) == 2 else y.ravel().reshape(-1, 1)
        self.y_predicted = None
        logging.debug("running %s with parameters:\n%s" % ( self.fun, "\n".join(["%s\t%s" % (kk, repr(vv)) for kk, vv in self.params.items() ]) ))
        self.rmodel = self._glmnet_(rmatrix(X), rmatrix(self.y), **self.params )

        if ("nfolds" in self.params) and (self.params["nfolds"] != 0):
            self.y_predicted = self["fit.preval"][:,:-1]
        if self.keep:
            #self.foldid=self["foldid"]
            self._foldid=self["foldid"]-1
        return self
    
    def predict(self, X, **kwargs):
        if "alpha" in kwargs:
            kwargs["s"] = kwargs.pop("alpha")
        elif "s" in kwargs:
            if type(kwargs["s"]) in (np.array, pd.Series, list, tuple):
                kwargs["s"] = FloatVector( kwargs["s"] )
        elif self.which_coef in ("1se", "min", "best"):
            kwargs["s"] = "lambda." + self.which_coef.replace("best", "min")

        if  ("nfolds" in self.params):
            predict = robjects.r("glmnet::predict.cv.glmnet")
        else:
            predict = robjects.r("glmnet::predict.glmnet")
        #logging.debug( repr(kwargs) )
        self.y_predicted = np.array(predict(self.rmodel, rmatrix(X), **kwargs))
        if (not hasattr(kwargs["s"], "__len__") or type(kwargs["s"]) is str ) and np.prod(self.y_predicted.shape) == X.shape[0]:
            self.y_predicted = self.y_predicted.ravel()
        return self.y_predicted
    
    @rplotwrap
    def rplot(self, file = None, **kwargs):
        rplot = robjects.r('plot')
        rplot(self.rmodel,  **kwargs)
        return
    
    def plot(self, x = "l1_norm", y = "coef", agg = None):
        if (x.lower() == "l1_norm") or (x.lower() == "l1 norm"):
            x_ = self.l1_norm
        elif x.lower().startswith("lambda") or x.lower().startswith("alpha"):
            x_ = self.alphas_
        else:
            raise ValueError("unknown argument for the x-axis:" % x)

        if y == "coef":
            tmp = self.which_coef
            self.which_coef = "all"
            y_ = self.coef[:,1:]
            self.which_coef = tmp
            ylab = "coefficients (without intercept)"
        elif y.lower().startswith("r2") or y.lower().startswith("r^2") :
            y_ = self.r2_path_.T
            ylab = "R^2"
        elif y.lower().startswith("mse"):
            y_ = self.mse_path_.T
            ylab = "MSE"
        elif (y.lower() == "l1_norm") or (y.lower() == "l1 norm"):
            y_ = self.l1_norm
            ylab = y 
        else:
            raise ValueError("unknown argument for the y-axis:", y)
        
        if type(agg) is str:
            agg_ = lambda x: np.__dict__[agg](x, axis = 1)

        if agg is not None:
            y_ = agg_(y_)

        try: 
            plt.plot( x_,  y_)
        except ValueError as ee:
            print( "x:", np.array(x_).shape, file = sys.stderr )
            print( "y:", np.array(y_).shape, file = sys.stderr )
            raise ee
        plt.xlabel(x)
        plt.ylabel( ylab )
        return plt.gcf()
    
    @property
    def fit_preval(self):
        y_chimeric = self["fit.preval"]
        return y_chimeric[:,~np.isnan(y_chimeric).any(axis=0)]

    @property
    def mse_path_(self, X = None):
        if hasattr(self, "_mse_path_"):
            return self._mse_path_

        if "nfolds"  in self.params and self.params["keep"]:
            ydiffsq = (self.fit_preval - self.y)**2
            self._mse_path_ = np.empty((ydiffsq.shape[1], self.nfolds))
            for nn in range(self.nfolds):
                self._mse_path_[:,nn] = ydiffsq[self._foldid==nn].mean(0)
            return  self._mse_path_

        logging.warn("well, this might be something strange; call it better on a CV object!")
        if self.y_predicted is None:
            logging.warning("No y_predicted has been found!")
            self.predict(X)
            #return None
            self._mse_path_ = self["cvm"]# (((self.y_predicted.T - self.y.ravel()).T)**2)
        if "nfolds" not in self.params:
            self._mse_path_ = self._mse_path_.mean(0)
        return  self._mse_path_

    def cross_val_score(self, metric, best = True):
        "apply supplied `metric(y, y_predicted)` to cached predicted y values"
        if not self.keep:
            raise ValueError("this method requires call with `keep=True` flag. " +\
                             "No cached values found.")
        if best:
            valid = self.alphas_ == self.alpha_
        else:
            valid = np.ones_like(self.alphas_, dtype = bool)

        metric_ = np.empty(self.nfolds)
        for nn in range(self.nfolds):
            fold_y_hat = self.fit_preval.T[ valid, self._foldid==nn].ravel()
            fold_y = self.y[self._foldid==nn].ravel()
            metric_[nn] = metric(fold_y, fold_y_hat)
        return metric_
 
    
    @property
    def r2_path_(self):
        if hasattr(self, "_r2_path_"):
            return self._r2_path_
        y_var_fold = np.empty((self.nfolds,))
        for nn in range(self.nfolds):
            #mse_path[nn] = ydiffsq[self.foldid==nn].mean(0)
            y_var_fold[nn] = self.y[self.foldid==nn].var()
        self._r2_path_ = 1- (self.mse_path_ / y_var_fold)
        return self._r2_path_ 
    
    @property
    def fit_preval(self):
        "cache to avoid frequent calls to R"
        if hasattr(self, "_fit_preval"):
            return self._fit_preval
        valid_alpha_inds = ~np.any(np.isnan(self["fit.preval"]), axis = 0)
        self._fit_preval = self["fit.preval"][:,valid_alpha_inds]
        return self._fit_preval
    
  #  @property
  #  def foldid(self):
  #      "fold id, zero-indexed"
  #      if hasattr(self, "_foldid"):
  #          return self._foldid
  #      #logging.debug("getting fold id")
  #      self._foldid = self.__getitem__( "foldid" ) - 1
  #      return self._foldid

    @property
    def alpha_(self):
        if self.cv is None:
            debug.warning("this is not a cross-validation object")
            return None
        if self.which_coef == "all" or ( self.which_coef == "1se"):
            return self["lambda.1se"][0]
        else:
            return self["lambda.min"][0]

    @property
    def coef(self):
        if ("nfolds" in self.params):
            coef = robjects.r('glmnet::coef.cv.glmnet')
            logging.debug("`which_coef` set to %s" % self.which_coef )
            if self.which_coef == "all":
                out = [ np.array(rmatrix(coef(self.rmodel, s = ss )))[:,0] for ss in self.alphas_ ]
            elif self.which_coef == "1se":
                out = np.array(rmatrix( coef(self.rmodel, s = self["lambda.1se"][0] ))).ravel()
            elif self.which_coef in ("best" , "min",  "lambda.min"):
                logging.debug("outputting best coefficients")
                out = np.array(rmatrix( coef(self.rmodel, s = self["lambda.min"][0] ))).ravel()
            else:
                raise ValueError("value of the attribute `which_coef` is wrong: %s" % self.which_coef )
        else:
            logging.debug("no `nfolds` key in `params`; outputting all coefficients")
            #out =  np.array(rmatrix( coef(self.rmodel) ))
            coef = robjects.r('glmnet::coef.glmnet')
            out = np.array(rmatrix( coef(self.rmodel )))#.ravel()
            return  out

        if type(out) is rpy2.robjects.methods.RS4:
            return np.array(rmatrix( out ))
        else:
            return np.array(out)

    @property
    def all_coef(self):
        tmp_which_coef = self.which_coef
        self.which_coef = "all"
        out = self.coef
        self.which_coef = tmp_which_coef
        return out

    @property
    def coef_(self):
        if ("nfolds" in self.params):
            coef = robjects.r('glmnet::coef.cv.glmnet')
            out = np.array(rmatrix( coef(self.rmodel, s = self["lambda.1se"][0] ))).ravel()[1:]
        else:
            coef = robjects.r('glmnet::coef.glmnet')
            out = np.array(rmatrix( coef(self.rmodel )))[1:].ravel()
        return out

    @property
    def nnz(self):
        """ Number of non-zero coefficients """
        return  (self.all_coef[:,1:] != 0).sum(1)
    
    @property
    def alphas_(self):
        outd = dict(self.rmodel.items())
        #valid_alpha_inds = ~np.any(np.isnan(self["fit.preval"]), axis = 0)
        return np.array(outd["lambda"])#[valid_alpha_inds]
    
    @property
    def l1_norm(self):
        return (abs(self.all_coef[:,1:])).sum(1)
    
    def __getattr__(self, attr):
        if attr == "lambda_":
            attr == "lambda"
        return self.__getitem__(attr)

    def  __getitem__(self, it):
        if "rmodel" not in self.__dict__:
            raise AttributeError("%r object has no attribute %r" %
                         (self.__class__, it))
            #return
        modeldict = dict(self.rmodel.items())
        if it in modeldict:
            x = modeldict[it]
        else:
            raise AttributeError("%r object has no attribute %r" %
                         (self.__class__, it))
        if type(x) in (FloatVector, IntVector):
            return np.array(x)
        if type(x) is Matrix:
            return np.array(x)
        if type(x) is rpy2.robjects.methods.RS4:
            try:
                return np.array(rmatrix(x))
            except:
                return x    
