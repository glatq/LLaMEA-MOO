"""MOEA/D-EGO building blocks, taken from DDMTOLab 1.0.10.

The start of the ablation path is MOEA/D-EGO as implemented in DDMTOLab
(``ddmtolab.Algorithms.STMO.MOEA_D_EGO``). Its helpers that do not depend on the
surrogate are imported and called unchanged. Three helpers call DDMTOLab's own
GP prediction directly, so they cannot be combined with another surrogate or
acquisition as they are: ``_get_estimation_z``, ``_moead_eti`` and the candidate
filter inside ``_opt_eti_fcm``. They are copied below with one change: the
prediction function (and, for the MOEA/D-DE search, the acquisition function)
is passed in as an argument. With DDMTOLab's own prediction and ETI passed in,
they execute the same operations in the same order as the originals, which the
tests check.

The copied code is from DDMTOLab (https://github.com/JiangtaoShen/DDMTOLab),
distributed under the MIT License:

    MIT License

    Copyright (c) 2025 Jiangtao Shen

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from types import SimpleNamespace

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm

L1 = 80  # DDMTOLab _L1: maximum points per local GP
L2 = 20  # DDMTOLab _L2: archive growth per extra cluster
NUM_WEIGHTS = {2: 200, 3: 210, 4: 295, 5: 456, 6: 462}  # DDMTOLab _opt_eti_fcm


def ddmtolab():
    """The DDMTOLab MOEA/D-EGO module (imported lazily: it pulls in torch/botorch)."""
    import ddmtolab.Algorithms.STMO.MOEA_D_EGO as mod

    return mod


# --------------------------------------------------------------------- unchanged
def initial_design(n, dim):
    """DDMTOLab's Latin hypercube initial design in the unit cube (``initialization``)."""
    from ddmtolab.Methods.Algo_Methods.algo_utils import initialization

    return initialization(SimpleNamespace(dims=[dim], n_tasks=1), [n], method="lhs")[0]


def weight_vectors(n_obj):
    """DDMTOLab's reference vectors for MOEA/D-EGO (as in ``_opt_eti_fcm``)."""
    from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point

    if n_obj <= 3:
        W, _ = uniform_point(NUM_WEIGHTS.get(n_obj, 200), n_obj, method="NBI")
    elif n_obj <= 6:
        W, _ = uniform_point(NUM_WEIGHTS.get(n_obj, 295), n_obj, method="ILD")
    else:
        W, _ = uniform_point(500, n_obj)
    return W


def remove_duplicates(new_decs, existing_decs):
    from ddmtolab.Methods.Algo_Methods.algo_utils import remove_duplicates as _rd

    return _rd(new_decs, existing_decs)


# ------------------------------------------------- copied, prediction passed in
def cluster_assignment(train_x, L1=L1, L2=L2):
    """The clustering of DDMTOLab's ``_build_gp_models_fcm``, without the GP fits.

    Returns ``(clusters, centers)``: the training-point indices of each local
    model and the cluster centres. With at most ``L1`` points there is one model
    on all of them.
    """
    from sklearn.cluster import KMeans

    K = train_x.shape[0]
    if K <= L1:
        return [np.arange(K)], train_x.mean(axis=0, keepdims=True)
    csize = 1 + int(np.ceil((K - L1) / L2))
    kmeans = KMeans(n_clusters=csize, n_init=10, random_state=0)
    kmeans.fit(train_x)
    centers = kmeans.cluster_centers_
    dis = cdist(train_x, centers)
    sorted_idx = np.argsort(dis, axis=0)
    return [sorted_idx[:L1, ci] for ci in range(csize)], centers


def estimate_ideal_point(D, predict_mean, ref_vecs, z_init):
    """DDMTOLab ``_get_estimation_z`` with the mean prediction passed in.

    ``predict_mean(X_unit) -> (N, M)`` replaces ``_gp_evaluate_mean_fcm``.
    """
    _operator_de = ddmtolab()._operator_de
    delta = 0.9
    nr = 2
    max_iter = 100
    pop_size = ref_vecs.shape[0]

    T = max(2, int(np.ceil(pop_size / 10)))
    B_dist = cdist(ref_vecs, ref_vecs)
    B = np.argsort(B_dist, axis=1)[:, :T]

    pop_x = np.random.rand(pop_size, D)
    pop_mean = predict_mean(pop_x)
    z = np.minimum(pop_mean.min(axis=0), z_init)

    for gen in range(max_iter - 1):
        for i in range(pop_size):
            if np.random.rand() < delta:
                P = B[i, np.random.permutation(T)]
            else:
                P = np.random.permutation(pop_size)

            off_x = _operator_de(pop_x[i], pop_x[P[0]], pop_x[P[1]])
            off_mean = predict_mean(off_x.reshape(1, -1)).flatten()

            z = np.minimum(z, off_mean)

            g_old = np.max((pop_mean[P] - z) * ref_vecs[P], axis=1)
            g_new = np.max(np.tile(off_mean - z, (len(P), 1)) * ref_vecs[P], axis=1)

            improve_idx = np.where(g_old > g_new)[0][:nr]
            if len(improve_idx) > 0:
                replace_idx = P[improve_idx]
                pop_x[replace_idx] = off_x
                pop_mean[replace_idx] = off_mean

    return z


def moead_search(D, predict, ref_vecs, gmin, acquisition):
    """DDMTOLab ``_moead_eti`` with the prediction and the acquisition passed in.

    ``predict(X_unit) -> (mean, std)``, each ``(N, M)``, replaces
    ``_gp_evaluate_fcm``; ``acquisition(mean, std, W, gmin) -> (N,)`` replaces
    ``_get_eti(mean, std, W, gmin, z)``. Returns the final population's
    acquisition values and positions (one solution per subproblem).
    """
    _operator_de = ddmtolab()._operator_de
    delta = 0.9
    nr = 2
    max_iter = 50
    pop_size = ref_vecs.shape[0]

    T = max(2, int(np.ceil(pop_size / 10)))
    B_dist = cdist(ref_vecs, ref_vecs)
    B = np.argsort(B_dist, axis=1)[:, :T]

    pop_x = np.random.rand(pop_size, D)
    pop_mean, pop_std = predict(pop_x)
    pop_eti = acquisition(pop_mean, pop_std, ref_vecs, gmin)

    for gen in range(max_iter - 1):
        for i in range(pop_size):
            if np.random.rand() < delta:
                P = B[i, np.random.permutation(T)]
            else:
                P = np.random.permutation(pop_size)

            off_x = _operator_de(pop_x[i], pop_x[P[0]], pop_x[P[1]])
            off_mean, off_std = predict(off_x.reshape(1, -1))
            off_mean = off_mean.flatten()
            off_std = off_std.flatten()

            eti_new = acquisition(
                np.tile(off_mean, (len(P), 1)),
                np.tile(off_std, (len(P), 1)),
                ref_vecs[P], gmin[P],
            )

            improve_idx = np.where(pop_eti[P] < eti_new)[0][:nr]
            if len(improve_idx) > 0:
                replace_idx = P[improve_idx]
                pop_x[replace_idx] = off_x
                pop_mean[replace_idx] = off_mean
                pop_std[replace_idx] = off_std
                pop_eti[replace_idx] = eti_new[improve_idx]

    return pop_eti, pop_x


def filter_candidates(candidate_x, cand_acq, train_x, D):
    """The candidate filter of DDMTOLab ``_opt_eti_fcm``.

    Keeps candidates with a positive acquisition value that are more than 1e-5
    from every evaluated point and every candidate kept before them.
    """
    Q = []
    Q_eti = []
    temp = train_x.copy()
    for idx in range(candidate_x.shape[0]):
        if candidate_x.shape[0] > 0 and temp.shape[0] > 0:
            min_dist = np.min(cdist(candidate_x[idx:idx + 1], temp))
        else:
            min_dist = np.inf
        if min_dist > 1e-5 and cand_acq[idx] > 0:
            Q.append(candidate_x[idx])
            Q_eti.append(cand_acq[idx])
            temp = np.vstack([temp, candidate_x[idx:idx + 1]])

    if len(Q) > 0:
        return np.array(Q), np.array(Q_eti)
    return np.empty((0, D)), np.empty(0)


# ------------------------------------------------------------- our extension
def augmented_tchebycheff_moments(u, sigma, ref_vecs, z, rho):
    """Mean and variance of the augmented Tchebycheff value under the GP (rule a).

    g = max_j a_j + rho * sum_j a_j with independent a_j ~ N(w_j (u_j - z_j),
    (w_j sigma_j)^2). Writing b_j = a_j + rho * sum_k a_k gives g = max_j b_j,
    whose terms are correlated, so Clark's moment matching is applied with the
    covariances (DDMTOLab's ``_app_max_of_2_gaussian`` assumes independence).
    The moments are exact for two objectives; for more, Clark's recursion is an
    approximation, as in DDMTOLab. With rho = 0 they are DDMTOLab's moments.
    """
    u = np.atleast_2d(u)
    sigma = np.atleast_2d(sigma)
    a_mu = ref_vecs * (u - z)
    a_var = (np.maximum(ref_vecs * sigma, 0)) ** 2
    n, m = a_mu.shape
    s_mu = a_mu.sum(axis=1)
    s_var = a_var.sum(axis=1)

    b_mu = a_mu + rho * s_mu[:, None]
    # cov(b_j, b_k) = delta_jk a_var_j + rho (a_var_j + a_var_k) + rho^2 s_var
    cov = (
        rho * (a_var[:, :, None] + a_var[:, None, :])
        + (rho ** 2) * s_var[:, None, None]
    )
    idx = np.arange(m)
    cov[:, idx, idx] += a_var

    m_y = b_mu[:, 0].copy()
    v_y = cov[:, 0, 0].copy()
    c_y = cov[:, 0, :].copy()  # cov(Y, b_l) for every l
    for k in range(1, m):
        v_k = cov[:, k, k]
        tau = np.sqrt(np.maximum(v_y + v_k - 2.0 * c_y[:, k], 1e-20))
        alpha = (m_y - b_mu[:, k]) / tau
        phi = norm.pdf(alpha)
        Phi = norm.cdf(alpha)
        Phi_n = norm.cdf(-alpha)
        new_m = m_y * Phi + b_mu[:, k] * Phi_n + tau * phi
        second = (
            (m_y ** 2 + v_y) * Phi
            + (b_mu[:, k] ** 2 + v_k) * Phi_n
            + (m_y + b_mu[:, k]) * tau * phi
        )
        c_y = c_y * Phi[:, None] + cov[:, k, :] * Phi_n[:, None]
        m_y = new_m
        v_y = np.maximum(second - new_m ** 2, 0)
    return m_y, v_y


def eti_augmented(u, sigma, ref_vecs, gbest, z, rho):
    """Expected improvement of the augmented Tchebycheff value, in DDMTOLab's ETI form."""
    m_y, v_y = augmented_tchebycheff_moments(u, sigma, ref_vecs, z, rho)
    std = np.sqrt(np.maximum(v_y, 1e-20))
    t = (gbest - m_y) / std
    return (gbest - m_y) * norm.cdf(t) + std * norm.pdf(t)
