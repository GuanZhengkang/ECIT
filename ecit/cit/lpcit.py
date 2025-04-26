import numpy as np
from scipy.linalg import sqrtm
import scipy.stats as stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel



def lpcit(x,y,z):
    return test_asymptotic_ci(x,y,z,rank=1000,alpha=0.05)['pvalue']


######



def sf_naka_p(J, x, p=1):
    mean = np.zeros(J)
    cov = np.eye(J)
    X = np.random.multivariate_normal(mean, cov, 1000000)
    S = np.sum(np.abs(X**p), 1)
    n = np.shape(S)[0]
    m = np.shape(S[S > x])[0]
    res = m / n
    return res


def gauss_kernel(X, test_locs, gwidth_2):
    n, d = X.shape
    X = X / np.sqrt(2 * gwidth_2)
    test_locs = test_locs / np.sqrt(2 * gwidth_2)
    D2 = (
        np.sum(X**2, 1)[:, np.newaxis]
        - 2 * np.dot(X, test_locs.T)
        + np.sum(test_locs**2, 1)
    )
    K = np.exp(-D2)
    return K


def compute_stat_ci(
    X,
    Y,
    Z,
    test_locs,
    p_norm,
    rank,
    rank_GP,
    gwidthX_2,
    gwidthY_2,
    gwidthZ_2,
    mu=1e-5,
    optimizer=False,
):
    n, dX = np.shape(X)
    n, dY = np.shape(Y)
    n, dZ = np.shape(Z)
    J, d_tot = np.shape(test_locs)

    rank_GP = min(rank_GP, 200)
    rank = min(n, rank)
    ind_r = np.random.choice(n, size=rank, replace=False)

    if rank_GP > rank:
        print("the number of samples for the GPR is bigger than the rank")
        rank_GP = rank

    test_locs_X = test_locs[:, :dX]
    KX_loc = gauss_kernel(X, test_locs_X, gwidthX_2)

    test_locs_Z = test_locs[:, dX + dY :]
    KZ_loc = gauss_kernel(Z, test_locs_Z, gwidthZ_2)

    KXZ_loc = KX_loc * KZ_loc
    KXZ_loc_r = KXZ_loc[ind_r, :]
    KXZ_loc_GP = KXZ_loc[ind_r[:rank_GP], :]

    test_locs_Y = test_locs[:, dX : dX + dY]
    KY_loc = gauss_kernel(Y, test_locs_Y, gwidthY_2)
    KY_loc_r = KY_loc[ind_r, :]
    KY_loc_GP = KY_loc[ind_r[:rank_GP], :]

    h_X = np.zeros((J, n))
    h_Y = np.zeros((J, n))

    lam_X, lam_Y, gwidthPredX, gwidthPredY = Initial_RLS_param(Z[ind_r, :], rank)

    for j in range(J):
        kernel_X = 1.0 * RBF(length_scale=gwidthPredX * np.ones(dZ)) + WhiteKernel(
            noise_level=lam_X
        )
        try:
            if optimizer == True:
                gp_X = GaussianProcessRegressor(kernel=kernel_X)
                gp_X.fit(Z[ind_r[:rank_GP], :], KXZ_loc_GP[:, j])
                kernel_X = gp_X.kernel_
                gp_X = GaussianProcessRegressor(kernel=kernel_X, optimizer=None)
                gp_X.fit(Z[ind_r, :], KXZ_loc_r[:, j])
            else:
                gp_X = GaussianProcessRegressor(kernel=kernel_X, optimizer=None)
                gp_X.fit(Z[ind_r, :], KXZ_loc_r[:, j])
        except ValueError:
            return "Error"

        kernel_Y = 1.0 * RBF(length_scale=gwidthPredY * np.ones(dZ)) + WhiteKernel(
            noise_level=lam_Y
        )
        try:
            if optimizer == True:
                gp_Y = GaussianProcessRegressor(kernel=kernel_Y)
                gp_Y.fit(Z[ind_r[:rank_GP], :], KY_loc_GP[:, j])
                kernel_Y = gp_Y.kernel_
                gp_Y = GaussianProcessRegressor(kernel=kernel_Y, optimizer=None)
                gp_Y.fit(Z[ind_r, :], KY_loc_r[:, j])
            else:
                gp_Y = GaussianProcessRegressor(kernel=kernel_Y, optimizer=None)
                gp_Y.fit(Z[ind_r, :], KY_loc_r[:, j])
        except ValueError:
            return "Error"

        h_X[j, :] = gp_X.predict(Z)
        h_Y[j, :] = gp_Y.predict(Z)

    U = (KXZ_loc - h_X.T) * (KY_loc - h_Y.T)
    S = np.mean(U, axis=0)
    Sigma = (1 / n) * np.dot(U.T, U)
    Sigma_mu = Sigma + mu * np.eye(J)
    Square_root = sqrtm(Sigma_mu)

    Normalized_S = np.linalg.solve(Square_root, S)
    res_NSS = (np.sqrt(n) ** p_norm) * np.sum((np.abs(Normalized_S)) ** p_norm)

    return res_NSS


def test_asymptotic_ci(
    X, Y, Z, rank, rank_GP=200, J=5, p_norm=2, mu=1e-10, alpha=0.05, optimizer=True
):
    test_locs, gwidthX_2, gwidthY_2, gwidthZ_2 = initial_T_gwidth2(
        X, Y, Z, n_test_locs=J
    )
    S = compute_stat_ci(
        X,
        Y,
        Z,
        test_locs,
        p_norm,
        rank,
        rank_GP,
        gwidthX_2,
        gwidthY_2,
        gwidthZ_2,
        mu=mu,
        optimizer=optimizer,
    )
    if S != "Error":
        if p_norm == 2:
            p_value = stats.chi2.sf(S, J)
        else:
            p_value = sf_naka_p(J, S, p=p_norm)

        results = {
            "alpha": alpha,
            "pvalue": p_value,
            "H0": p_value > alpha,
            "test statistic": S,
        }

        return results

    else:
        return "Error"


def dist_matrix(X, Y):
    sx = np.sum(X**2, 1)
    sy = np.sum(Y**2, 1)
    D2 = sx[:, np.newaxis] - 2.0 * np.dot(X, Y.T) + sy[np.newaxis, :]
    D2[D2 < 0] = 0
    D = np.sqrt(D2)
    return D


def meddistance(X, subsample=None, mean_on_fail=True):
    if subsample is None:
        D = dist_matrix(X, X)
        Itri = np.tril_indices(D.shape[0], -1)
        Tri = D[Itri]
        med = np.median(Tri)
        if med <= 0:
            # use the mean
            return np.mean(Tri)
        return med

    else:
        assert subsample > 0
        rand_state = np.random.get_state()
        n = X.shape[0]
        ind = np.random.choice(n, min(subsample, n), replace=False)
        np.random.set_state(rand_state)
        # recursion just one
        return meddistance(X[ind, :], None, mean_on_fail)


def initial_T_gwidth2(X, Y, Z, n_test_locs=5):
    n, dX = X.shape
    n, dY = Y.shape
    n, dZ = Z.shape
    J = n_test_locs

    mean_x = np.mean(X, 0)
    mean_y = np.mean(Y, 0)
    mean_z = np.mean(Z, 0)
    cov_x = np.cov(X.T)
    [Dx, Vx] = np.linalg.eig(cov_x + 1e-3 * np.eye(dX))
    Dx = np.real(Dx)
    Vx = np.real(Vx)
    Dx[Dx <= 0] = 1e-3
    eig_pow = 0.9
    reduced_cov_x = Vx.dot(np.diag(Dx**eig_pow)).dot(Vx.T) + 1e-3 * np.eye(dX)

    cov_y = np.cov(Y.T)
    [Dy, Vy] = np.linalg.eig(cov_y + 1e-3 * np.eye(dY))
    Vy = np.real(Vy)
    Dy = np.real(Dy)
    Dy[Dy <= 0] = 1e-3
    reduced_cov_y = Vy.dot(np.diag(Dy**eig_pow).dot(Vy.T)) + 1e-3 * np.eye(dY)

    cov_z = np.cov(Z.T)
    [Dz, Vz] = np.linalg.eig(cov_z + 1e-3 * np.eye(dZ))
    Vz = np.real(Vz)
    Dz = np.real(Dz)
    Dz[Dz <= 0] = 1e-3
    reduced_cov_z = Vz.dot(np.diag(Dz**eig_pow).dot(Vz.T)) + 1e-3 * np.eye(dZ)

    Tx = np.random.multivariate_normal(mean_x, reduced_cov_x, J)
    Ty = np.random.multivariate_normal(mean_y, reduced_cov_y, J)
    Tz = np.random.multivariate_normal(mean_z, reduced_cov_z, J)
    T0 = np.hstack((Tx, Ty, Tz))

    med_X = meddistance(X, 1000)
    gwidthX_2 = med_X**2

    med_Y = meddistance(Y, 1000)
    gwidthY_2 = med_Y**2

    med_Z = meddistance(Z, 1000)
    gwidthZ_2 = med_Z**2

    return (T0, gwidthX_2, gwidthY_2, gwidthZ_2)


def Initial_RLS_param(Z, rank):
    lam_X, lam_Y = rank ** (-1 / 1.5), rank ** (-1 / 1.5)
    med_Z = meddistance(Z, 1000)
    gwidthPredX, gwidthPredY = med_Z, med_Z

    return lam_X, lam_Y, gwidthPredX, gwidthPredY
