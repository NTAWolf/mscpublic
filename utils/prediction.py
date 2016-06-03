"""
Tools for building and evaluating models
"""

import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

import tbtools.iter as tbiter
import tbtools.panda as tbpd
import tbtools.func as tbfunc


def rmse(resid):
    return np.sqrt(np.mean(np.square(resid)))


def fit_standardizer(x):
    """Fits a standardizer to x, and returns
    it as a function that directly transforms
    the known columns of a given x.
    """
    trmean = x.mean()
    trvar = x.std()
    trmean['intercept'] = 0
    trvar['intercept'] = 1

    def standardize(x2):
        return (x2-trmean[x2.columns])/trvar[x2.columns]

    return standardize


def get_residuals(model, y, x, pred_transform=None):
    pred = get_prediction(model, x, pred_transform)
    resid = pd.Series(y - pred, index=y.index, name='Residuals')
    return resid


def get_prediction(model, x, pred_transform):
    pred = model.predict(x)
    if pred_transform is not None:
        pred = pred_transform(pred)
    return pred


def score_model(residuals, y):
    print('RMSE: {:.3f}'.format(rmse(resid)))
    print('RMSE: {:.2f} for target > 0'.format(rmse(resid[y != 0])))
    for p in (50, 70, 80, 90, 95, 99):
        v = np.percentile(y, p)
        print('RMSE: {:.3f} for target > {} (top {} percentile)'.format(
            rmse(resid[y > v]), v, 100-p))


def plot_model(prediction, y, x):
    fig, axs = sns.plt.subplots(2, 2, figsize=(16, 10))
    axs = axs.flatten()

    resid = pd.Series(y - prediction, index=y.index, name='Residuals')

    resid.hist(bins=40, ax=axs[0])
    axs[0].set_xlabel('Residuals')
    sm.qqplot(resid, line='q', ax=axs[1])
    axs[1].set_xlabel('Residuals')
    tbpd.hist2d(resid, prediction, ax=axs[2],
                vlabel='Residuals', hlabel='Predicted value',
                integer_aligned_bins=True)
    tbpd.hist2d(y, prediction, ax=axs[3],
                vlabel='True value', hlabel='Predicted value',
                integer_aligned_bins=True, sqrt=True)
    fig.tight_layout()


def plot_fit(target, prediction,
             target_name=None, prediction_name=None,
             axs=None):
    if axs is None:
        fig, axs = sns.plt.subplots(1, 2, figsize=(14, 7))
        axs = axs.flatten()

    resid = target - prediction

    if target_name is None:
        if hasattr(target, 'name'):
            target_name = target.name
        else:
            target_name = 'Target'

    if prediction_name is None:
        if hasattr(prediction, 'name'):
            prediction_name = prediction.name
        else:
            prediction_name = 'Prediction'

    tbpd.hist2d(v=resid, h=prediction, vlabel='Residuals',
                hlabel=prediction_name, integer_aligned_bins=True,
                ax=axs[0])
    tbpd.hist2d(v=resid, h=target, vlabel='Residuals',
                hlabel=target_name, integer_aligned_bins=True,
                ax=axs[1])


def variable_residuals(resid, varname, data, ax=None):
    tbpd.hist2d(v=resid, h=data[varname], vlabel='Residuals',
                hlabel=varname, integer_aligned_bins=True,
                ax=ax)


def within(resid, percentile=80):
    ar = np.abs(resid)
    n = np.percentile(ar, percentile)
    return n


class ResultsAggregator:

    def __init__(self, data=None):
        """
        data is a dict that can be indexed as
            data['train']['x']
        for all data sets
        """
        if data is not None:
            self.update_data(data)
        self.results = {}

    def update_data(self, data):
        """
        data is a dict that can be indexed as
            data['train']['x']
        for all data sets
        """
        self.data = data

    def _insert_results(self, name, performance, residuals, model):
        self.results[name] = {'performance': performance,
                              'residuals': residuals,
                              # 'model': model
                              }

    def append(self, name, model_builder):
        performance = {}
        residuals = {}
        model = model_builder().fit(self.data['train']['x'],
                                    self.data['train']['y'])

        for splitname, split_ in self.data.items():
            x, y = split_['x'], split_['y']
            pred = model.predict(x)
            resid = y - pred
            residuals[splitname] = resid
            performance[
                'rmse ' + splitname] = rmse(resid)
            performance['within_80% ' + splitname] = within(resid, 80)

        # CV for Standard Error
        self._standard_error(model_builder, performance)

        self._insert_results(
            name, performance, residuals, model)

    def to_df(self):
        """
        Returns a pandas dataframe representation of the performance results
        stored within.
        """
        perfs = {name: data['performance']
                 for name, data in self.results.items()}
        df = pd.DataFrame(perfs).T.sort_index()
        df.columns = df.columns.str.split(expand=True)
        return df

    def plot_residuals(self, name):
        """
        Crude plotting of residuals vs target
        """
        r = self.results[name]['residuals']
        fig, axs = sns.plt.subplots(1, len(self.data), figsize=(14, 6))
        axs = axs.flatten()

        for ax, (split, data) in zip(axs, self.data.items()):
            resid = r[split]
            tbpd.hist2d(v=data['y'], h=r[split], vlabel='Target',
                        hlabel='Residuals ({})'.format(split),
                        integer_aligned_bins=True,
                        ax=ax)

    def _standard_error(self, model_builder, performance):
        """Calculates standard error of the mean of
        RMSE and within_80% estimates, and stores
        the results in performance

        model_builder is a function that takes no arguments, and returns
            a new model. The model can be subjected to .fit(x,y) and .predict(x).
            The model is expected to handle transformations of x and y
        performance is a dict to be stored in self.results
        """

        y = self.data['train']['y']
        x = self.data['train']['x']
        train_days = np.unique(y.index.date)

        rmses = np.zeros(len(train_days))
        w80s = np.zeros(len(train_days))

        for i in range(len(train_days)):
            holdout = np.in1d(y.index.date, train_days[i])
            build = y.index[~holdout]
            holdout = y.index[holdout]

            model = model_builder()
            model = model.fit(x=x.loc[build],
                              y=y.loc[build])

            resid = y.loc[holdout] - model.predict(x.loc[holdout])

            w80s[i] = within(resid, 80)
            rmses[i] = rmse(resid)

        performance['rmse train_cv_mean'] = np.mean(rmses)
        performance['rmse train_cv_sem'] = stats.sem(rmses)
        performance['within_80% train_cv_mean'] = np.mean(w80s)
        performance['within_80% train_cv_sem'] = stats.sem(w80s)


class PersistentResultsAggregator(ResultsAggregator):

    def __init__(self, path, data=None):
        """
        path is an absolute path on your file system.
            This is where results are loaded from and written to.
        data is a dict that can be indexed as
            data['train']['x']
        for all data sets
        """
        self.path = path
        super().__init__(data)
        # Read results from disk
        if os.path.isfile(path):
            # print('Results already exist in', path)
            # while True:
            #     v = input('Delete them (d) or append to them (a)? >')
            #     if v == 'd':
            #         os.remove(path)
            #         break
            #     elif v == 'a':
            with open(path, 'rb') as f:
                self.results = pickle.load(f)
            print('Appeding to existing results')
                    # break

    def _insert_results(self, *args, **kwargs):
        super()._insert_results(*args, **kwargs)
        with open(self.path, 'wb') as f:
            pickle.dump(self.results, f)


# Model builders

class LinRegWrapper:

    """Handles sqrt transform and standardization,
    and drops columns of all 0s in the training set.
    """

    def __init__(self, subset=None):
        self.subset = subset

    def _subset(self, x):
        if self.subset is not None:
            x = x[self.subset]
        return x

    def fit(self, x, y):
        x = self._subset(x)
        # Update subset to also drop all-0 columns
        self.subset = x.columns[~(x==0).all()]
        x = self._subset(x)
        self.standardize = fit_standardizer(x)
        x = self.standardize(x)
        x['intercept'] = 1

        # Drop insignificant features
        for i in range(x.shape[1]):
            res = sm.OLS(np.sqrt(y), x).fit()
            if res.pvalues.max() > .05:
                c = res.pvalues.argmax()
                x = x.drop(c, axis=1)
            else:
                break
        self.model = res
        self.subset = x.columns
        self.pvalues = res.pvalues
        return self

    def predict(self, x):
        x = self.standardize(self._subset(x))
        return np.square(self.model.predict(x))


class RFWrapper:

    def __init__(self, subset=None, ntrees=200):
        self.subset = subset
        self.ntrees = ntrees

    def _subset(self, x):
        if self.subset is not None:
            x = x[self.subset]
        return x

    def fit(self, x, y):
        x = self._subset(x)
        res = RandomForestRegressor(n_estimators=self.ntrees,
                                    n_jobs=-1).fit(x, y)
        self.model = res
        return self

    def predict(self, x):
        x = self._subset(x)
        return self.model.predict(x)

class BaselineMean:
    def __init__(self):
        pass

    def fit(self, x, y):
        self.mean = np.mean(y)
        return self

    def predict(self, x):
        return self.mean

class BaselineLastC:
    def __init__(self):
        pass

    def fit(self, x, y):
        return self

    def predict(self, x):
        cval = [v for v in x if v.startswith('C ')][0]
        return x[cval]

_kind2class = {
    'rf':RFWrapper,
    'lr':LinRegWrapper,
    'bm':BaselineMean,
    'bc':BaselineLastC,
}

def model_builder_builder(kind, *args, **kwargs):
    """Yes, this builds model builders.

    kind is
        'rf' for random forest
        'lr' for linear regression
        'bm' for baseline: mean of training set C
        'bc' for baseline: last seen value of C
    args and kwargs are passed to the respective
        RFWrapper and
        LinRegWrapper's init methods.
    """
    cl = _kind2class[kind]

    def model_builder():
        return cl(*args, **kwargs)

    return model_builder
