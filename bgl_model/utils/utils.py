from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import norm
from scipy.optimize import lsq_linear
import GPy

def std_bolus_calc(cho, gm, gt, cr, cf, iob=None):
    _iob = iob if iob else 0
    return np.max([0, cho / cr + (gm - gt) / cf - _iob])


def scale(X, scalers):

    if type(X) in {float, int, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64} and type(scalers)==StandardScaler:
        X_scaled = scalers.transform(np.atleast_2d([X]).T)[0][0]

    elif type(X)==np.ndarray and len(X.shape)==1 and type(scalers)==StandardScaler:
        X_scaled = scalers.transform(np.atleast_2d(X).T).reshape(-1)

    elif type(X)==np.ndarray and len(X.shape)>1 and type(scalers)==list:
        X_scaled = np.zeros(X.shape)
        for d in range(X.shape[1]):
            X_scaled[:,d] = scalers[d].transform(np.atleast_2d(X[:,d]).T)[:,0] if scalers[d] else X[:,d]
    else:
        raise ValueError('Improper usage of scale, please check the source code')

    return X_scaled


def unscale(X, scalers):

    if type(X) in {float, int, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64} and type(scalers)==StandardScaler:
        X_unscaled = scalers.inverse_transform(np.atleast_2d([X]).T)[0][0]

    elif type(X)==np.ndarray and len(X.shape)==1 and type(scalers)==StandardScaler:
        X_unscaled = scalers.inverse_transform(np.atleast_2d(X).T, copy=True).reshape(-1)

    elif type(X)==np.ndarray and len(X.shape)>1 and type(scalers)==list:
        X_unscaled = np.zeros(X.shape)
        for d in range(X.shape[1]):
            X_unscaled[:,d] = scalers[d].inverse_transform(np.atleast_2d(X[:,d]).T, copy=True)[:,0] if scalers[d] else X[:,d]

    else:
        raise ValueError('Improper usage of unscale, please check the source code')

    return X_unscaled



def scale_context(context, variables, scalers):

    if context is None:
        return dict()

    pcontext = dict()
    for i in range(len(variables)):
        vname, vscaler = variables[i]['name'], scalers[i]
        if vname not in context.keys():
            continue
        val = context[vname]
        pcontext[vname] = scale(val, vscaler) if vscaler else val
    return pcontext


def get_feasible_points(context, domains, nsamples=None):

    I_x = np.ones(nsamples)
    ins_min, ins_max = domains['insulin']
    x = np.ones((nsamples, 2))
    x[:, 0] = context['meal']
    x[:, 1] = np.array([round(v, 2) for v in np.linspace(ins_min, ins_max, nsamples)])

    return (x[:,1][I_x.astype(bool)], x[:,1])


def create_gp(X, y, mean_func, noise_var, normalize_Y):

    kern_gp = GPy.kern.Matern52(input_dim=X.shape[1], ARD=True, lengthscale=1.5) + \
                GPy.kern.Linear(input_dim=X.shape[1], ARD=True)

    model_gp = GPy.models.GPRegression(X, y,
                                        kernel=kern_gp,
                                        noise_var=noise_var,
                                        mean_function=mean_func,
                                        normalizer=normalize_Y)

    return model_gp
