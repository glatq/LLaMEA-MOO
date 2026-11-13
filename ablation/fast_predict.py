"""Fast single-point GP predictions for the MOEA/D inner loops of the S blocks.

DDMTOLab's MOEA/D-EGO estimates the ideal point and searches for candidates with a
steady-state MOEA/D: each offspring is predicted on its own before the next one is
made, about 30,000 predictions per iteration. With BoTorch, each prediction costs
about 1 ms of call overhead, which is about 95% of a run. The predictors here compute
the same posterior in closed form with numpy, from the parameters of the models
already fitted. They fit nothing and change no hyperparameter. They agree with the
original predictions to the original's float32 rounding (BoTorch) or to float64
rounding (scikit-learn); tests/test_ablation_framework.py checks both.

``ModularMOEAD(fast_predict=True)`` uses them only inside those two inner loops
(blocks 7 and 9 as S). Every other prediction, and every T block, uses the
original model, so ``TTTTTTTTTT`` still reproduces MOEAD-EI Hybrid exactly.
"""

import numpy as np
from scipy.spatial.distance import cdist

# gpytorch clamps a float32 posterior variance to at least this value
# (gpytorch.settings.min_variance), after BoTorch's Standardize has scaled it back.
_GPYTORCH_MIN_VARIANCE_FLOAT32 = 1e-6


# ------------------------------------------------------------------- BoTorch
def _botorch_parts(gp):
    """Closed-form pieces of a fitted DDMTOLab ``gp_build`` model, or None."""
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.kernels import RBFKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.means import ConstantMean

    if not (
        type(gp.covar_module) is RBFKernel
        and type(gp.mean_module) is ConstantMean
        and type(gp.likelihood) is GaussianLikelihood
        and isinstance(getattr(gp, "outcome_transform", None), Standardize)
        and getattr(gp, "input_transform", None) is None
    ):
        return None
    X = gp.train_inputs[0].detach().double().numpy()
    y = gp.train_targets.detach().double().numpy()  # standardised
    ls = gp.covar_module.lengthscale.detach().double().numpy().reshape(-1)
    noise = float(gp.likelihood.noise.detach().double().reshape(-1)[0])
    c = float(gp.mean_module.constant.detach().double().reshape(-1)[0])
    ot = gp.outcome_transform
    y_mean = float(ot.means.detach().double().reshape(-1)[0])
    y_std = float(ot.stdvs.detach().double().reshape(-1)[0])
    Xs = X / ls
    sq = (Xs**2).sum(1)
    K = np.exp(-0.5 * np.maximum(sq[:, None] + sq[None, :] - 2 * Xs @ Xs.T, 0.0))
    K[np.diag_indices_from(K)] += noise
    Linv = np.linalg.solve(np.linalg.cholesky(K), np.eye(len(X)))
    alpha = Linv.T @ (Linv @ (y - c))
    return Xs, sq, 1.0 / ls, alpha, Linv, c, y_mean, y_std


def ddmtolab_fcm_predictor(models, centers):
    """Fast equivalent of DDMTOLab's ``_gp_evaluate_fcm(X, models, centers)``.

    ``models[cluster][objective]`` are the fitted ``gp_build`` models and
    ``centers`` their cluster centres; each point is predicted by the models of
    its nearest centre. Returns ``(mean, std)`` in the objectives' own sign and
    scale, as the original does. None if a model is not of the expected form.
    """
    clusters = []
    for cluster in models:
        parts = [_botorch_parts(gp) for gp in cluster]
        if any(p is None for p in parts):
            return None
        clusters.append([np.stack(field) for field in zip(*parts)])  # stacked over objectives
    centers = np.asarray(centers, dtype=float)

    def predict_one(x):
        """One point, as the inner loops call it: the same arithmetic, fewer calls."""
        ci = int(np.argmin(cdist(x[None, :], centers)[0])) if len(clusters) > 1 else 0
        Xs, sq, inv_ls, alpha, Linv, c, y_mean, y_std = clusters[ci]
        z = x * inv_ls  # (objectives, dim)
        r2 = (z * z).sum(1)[:, None] + sq - 2 * np.einsum("md,mnd->mn", z, Xs)
        k = np.exp(-0.5 * np.maximum(r2, 0.0))
        v = np.einsum("mn,mjn->mj", k, Linv)
        var = np.maximum((1.0 - (v * v).sum(1)) * y_std**2, _GPYTORCH_MIN_VARIANCE_FLOAT32)
        mean = c + (k * alpha).sum(1)
        return -(mean * y_std + y_mean)[None, :], np.sqrt(var)[None, :]

    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.shape[0] == 1:
            return predict_one(X[0])
        nearest = np.argmin(cdist(X, centers), axis=1)  # as DDMTOLab
        n_obj = clusters[0][0].shape[0]
        u = np.empty((X.shape[0], n_obj))
        s = np.empty((X.shape[0], n_obj))
        for ci in np.unique(nearest):
            Xs, sq, inv_ls, alpha, Linv, c, y_mean, y_std = clusters[ci]
            mask = nearest == ci
            Z = X[mask][None, :, :] * inv_ls[:, None, :]  # (objectives, points, dim)
            r2 = (Z**2).sum(2)[:, :, None] + sq[:, None, :] - 2 * np.einsum("mtd,mnd->mtn", Z, Xs)
            k = np.exp(-0.5 * np.maximum(r2, 0.0))
            mean = c[:, None] + np.einsum("mtn,mn->mt", k, alpha)
            v = np.einsum("mtn,mjn->mtj", k, Linv)
            var = (1.0 - (v * v).sum(2)) * (y_std**2)[:, None]
            var = np.maximum(var, _GPYTORCH_MIN_VARIANCE_FLOAT32)
            # gp_build models -objs, and gp_predict negates the mean back
            u[mask] = -(mean * y_std[:, None] + y_mean[:, None]).T
            s[mask] = np.sqrt(var).T
        return u, s

    return predict


# -------------------------------------------------------------- scikit-learn
def _matern(r, nu):
    if nu == 0.5:
        return np.exp(-r)
    if nu == 1.5:
        a = np.sqrt(3.0) * r
        return (1.0 + a) * np.exp(-a)
    if nu == 2.5:
        a = np.sqrt(5.0) * r
        return (1.0 + a + a * a / 3.0) * np.exp(-a)
    return None


def _sklearn_parts(gp):
    """Closed-form pieces of MOEAD-EI Hybrid's fitted GaussianProcessRegressor, or None.

    The Hybrid's kernels are ``C * Matern + White`` or, in its fallback,
    ``Matern + White``.
    """
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, Product, Sum, WhiteKernel

    k = gp.kernel_
    if not (isinstance(k, Sum) and isinstance(k.k2, WhiteKernel)):
        return None
    white, main = float(k.k2.noise_level), k.k1
    if isinstance(main, Product) and isinstance(main.k1, ConstantKernel) and isinstance(main.k2, Matern):
        amp, mat = float(main.k1.constant_value), main.k2
    elif isinstance(main, Matern):
        amp, mat = 1.0, main
    else:
        return None
    if _matern(np.zeros(1), mat.nu) is None:
        return None
    ls = np.broadcast_to(np.asarray(mat.length_scale, dtype=float), (gp.X_train_.shape[1],))
    Linv = np.linalg.solve(gp.L_, np.eye(gp.L_.shape[0]))
    y_mean = float(np.ravel(gp._y_train_mean)[0])
    y_std = float(np.ravel(gp._y_train_std)[0])
    return gp.X_train_ / ls, 1.0 / ls, amp, float(mat.nu), white, gp.alpha_.ravel(), Linv, y_mean, y_std


def hybrid_predictor(scaler_X, models):
    """Fast equivalent of MOEAD-EI Hybrid's prediction: ``scaler_X`` then each GP's
    ``predict(return_std=True)``, stacked over objectives. None if a model is not of
    the expected form."""
    parts = [_sklearn_parts(gp) for gp in models]
    if any(p is None for p in parts):
        return None
    mean_, scale_ = np.asarray(scaler_X.mean_, dtype=float), np.asarray(scaler_X.scale_, dtype=float)

    def predict(Xr):
        Xs = (np.asarray(Xr, dtype=float) - mean_) / scale_
        mus, sds = [], []
        for Xt, inv_ls, amp, nu, white, alpha, Linv, y_mean, y_std in parts:
            # distances from differences, as scikit-learn's cdist: exp(-r) is not
            # smooth at r = 0, where the expanded square would lose precision
            D = (Xs * inv_ls)[:, None, :] - Xt[None, :, :]
            k = amp * _matern(np.sqrt((D * D).sum(2)), nu)
            v = k @ Linv.T
            var = np.maximum(amp + white - (v * v).sum(1), 0.0)
            mus.append(y_std * (k @ alpha) + y_mean)
            sds.append(np.sqrt(var) * y_std)
        return np.array(mus).T, np.array(sds).T

    return predict
