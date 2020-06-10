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
    c[c < 0.] = 0.

    K = (np.ones(shape=M) * mu_K) if np.isclose(sigma_K, 0.) else np.random.normal(loc=mu_K, scale=sigma_K, size=M)
    m = (np.ones(shape=S) * mu_m) if np.isclose(sigma_m, 0.) else np.random.normal(loc=mu_m, scale=sigma_m, size=S)

    alpha_d = draw_alpha_d_matix(S, mu_d, sigma_d, gamma)
    omega = calc_omega(MS_ratio, sigma_c, sigma_d, mat_norm_ratio)
    alpha_d *= omega

    return c, alpha_d, m, K


def get_f(c, alpha_d, m, K, kappa=0., migration=0.):
    A = np.vstack([np.hstack([alpha_d, -c]), np.hstack([c.T, np.eye(c.shape[1])])])
    B = np.concatenate([-m, K])
    S = m.shape[0]

    def f(t, psi):
        d_psi = psi * (B - np.matmul(A, psi)) + migration
        d_psi[:S] += - psi[:S] * (psi[:S] ** 2. * kappa)
        return d_psi

    return f


def get_jec(c, alpha_d, m, K, kappa=0.):
    A = np.vstack([np.hstack([alpha_d, -c]), np.hstack([c.T, np.eye(c.shape[1])])])
    B = np.concatenate([-m, K])
    S = m.shape[0]

    def jac(t, psi):
        j = np.diag(B - np.matmul(A, psi)) - (A * psi[:, np.newaxis])
        j[:S, :S] += - np.diag(3. * psi[:S] ** 2. * kappa)
        return j

    return jac


def integrate(c, alpha_d, m, K, t_span, kappa=0., migration=1e-10, psi0=None, num_time_evals=1000):
    S = m.shape[0]
    M = K.shape[0]
    t_eval = None if (num_time_evals is None) else np.linspace(0., t_span, num=num_time_evals)
    psi0 = np.random.rand(S + M) if (psi0 is None) else psi0

    res = solve_ivp(fun=get_f(c, alpha_d, m, K, kappa, migration),
                    t_span=(0, t_span),
                    t_eval=t_eval,
                    y0=psi0,
                    atol=migration / 100.,
                    rtol=1e-8,
                    method='RK45')
                    #method='Radau', jac=(c, alpha_d, m, K, kappa))

    N = res.y[:S, :]
    R = res.y[S:, :]
    return res.t, N, R


def run_full_mcrm(S, M, mu_c, sigma_c, mu_d, sigma_d, gamma, mu_K, sigma_K, mu_m, sigma_m, mat_norm_ratio,
                  t_span, kappa, num_time_evals=1000):

    c, alpha_d, m, K = draw_interactions(S, M, mu_c, sigma_c, mu_d, sigma_d,
                                         gamma, mu_K, sigma_K, mu_m, sigma_m, mat_norm_ratio)

    return integrate(c, alpha_d, m, K, t_span, kappa, num_time_evals=num_time_evals)
