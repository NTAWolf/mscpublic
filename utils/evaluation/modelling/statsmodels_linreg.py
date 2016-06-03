import statsmodels.api as sm
from . import rmse, get_residuals, get_prediction, score_model, plot_model

# Make it super easy to run and evaluate linear models


def reduce(y, x, cols=None, add_intercept=True,
           remove_mean=True, divide_std=True,
           target_transform=None, target_inverse_transform=None):
    with tbdev.Notify('Reduction done'):
        x = x.copy()
        if remove_mean:
            x = x - x.mean()
        if divide_std:
            x = x/x.std()
        if add_intercept:
            x['intercept'] = 1

        model, x = reduce_until_significant(y, x,
                                            target_transform,
                                            target_inverse_transform)
        print('\n\n\n')
        score_model(get_residuals(model, y, x, target_inverse_transform), y)
        plot_model(get_prediction(model, x, target_inverse_transform), y, x)

    return model

def reduce_until_significant(y, x, target_transform, target_inverse_transform):
    drops = []
    rmses = []
    pb = tbiter.IProgressBar(range(len(x.columns)))
    if target_transform is not None:
        yt = target_transform(y)
    else:
        yt = y

    for i in pb:
        res = sm.OLS(yt, x).fit()
        e = rmse(res.resid)
        rmses.append(e)
        if res.pvalues.max() > .05:
            c = res.pvalues.argmax()
            pb.set_state('RMSE={:.2f} p={:.2f} "{}"'.format(e, res.pvalues[c], c))
            drops.append(c)
            x = x.drop(c, axis=1)
        else:
            pb.set_state('Finished.')
            pb.finish()
            print('Stopping at {} drops'.format(len(drops)))
            break
    print(res.summary())
    return res, x