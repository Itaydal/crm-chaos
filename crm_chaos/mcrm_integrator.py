import numpy as np
from scipy.integrate import solve_ivp


def draw_alpha_d_matix(S, mu, sigma, gamma):
    alpha_d_diag = np.random.normal(loc=mu/S, scale=sigma/np.sqrt(S), size=S)
    off_diag_pair_cov = np.array([[1., gamma], [gamma, 1.]]) * sigma**2. / S
    alpha_d_off_diag = np.random.multivariate_normal(mean=np.array([mu/S, mu/S]), cov=off_diag_pair_cov, size=int(S*(S-1)/2))

    alpha_d = np.zeros(shape=(S, S))
    alpha_d[np.diag_indices(S)] = alpha_d_diag
    alpha_d[np.triu_indices(S, 1)] = alpha_d_off_diag[:, 0]
    alpha_d = alpha_d.T
    alpha_d[np.triu_indices(S, 1)] = alpha_d_off_diag[:, 1]
    return alpha_d


def calc_omega(MS_ratio, sigma_c, sigma_alpha, mat_norm_ratio):
    return mat_norm_ratio * np.sqrt((MS_ratio + 1.) * MS_ratio) * (sigma_c ** 2.) / sigma_alpha


def draw_interactions(S, M, mu_c, sigma_c, mu_d, sigma_d, gamma, mu_K, sigma_K, mu_m, sigma_m, mat_norm_ratio):
    MS_ratio = 1. * M / S

    c = np.random.normal(loc=mu_c / S, scale=sigma_c / np.sqrt(S), size=[S, M])
    K = (np.ones(shape=M) * mu_K) if np.isclose(sigma_K, 0.) else np.random.normal(loc=mu_K, scale=sigma_K, size=M)
    m = (np.ones(shape=S) * mu_m) if np.isclose(sigma_m, 0.) else np.random.normal(loc=mu_m, scale=sigma_m, size=S)

    alpha_d = draw_alpha_d_matix(S, mu_d, sigma_d, gamma)
    omega = calc_omega(MS_ratio, sigma_c, sigma_d, mat_norm_ratio)

    alpha = np.dot(c, c.T) + (omega * alpha_d)
    K_eff = np.dot(c, K) - m

    return alpha, K_eff


def get_f(alpha, K, migration):
    def f(t, N):
        return N * (K - np.dot(alpha, N)) + migration
    return f


def get_jac(alpha, K):
    def jac(t, N):
        return np.diag(K - np.dot(alpha, N)) - (alpha * N[:, np.newaxis])
    return jac


def integrate(alpha, K, t_span, migration=1e-10, N0=None, num_samples=1000):
    S = alpha.shape[0]
    t_eval = np.linspace(0., t_span, num=num_samples)
    N0 = np.random.rand(S) if (N0 is None) else N0

    res = solve_ivp(fun=get_f(alpha, K, migration),
                    t_span=(0, t_span),
                    t_eval=t_eval,
                    y0=N0,
                    atol=migration / 10.,
                    method='Radau', jac=get_jac(alpha, K))

    return res.t, res.y


def run_mcrm(S, M, mu_c, sigma_c, mu_d, sigma_d, gamma,
             mu_K, sigma_K, mu_m, sigma_m, mat_norm_ratio, t_span):
    alpha, K = draw_interactions(S, M, mu_c, sigma_c, mu_d, sigma_d, gamma, mu_K, sigma_K, mu_m, sigma_m, mat_norm_ratio)
    return integrate(alpha, K, t_span)
