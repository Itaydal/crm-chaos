import numpy as np
from scipy.integrate import solve_ivp


def draw_interactions(S, M, mu_c, sigma_c, mu_K, sigma_K, mu_m, sigma_m):
    c = np.random.normal(loc=mu_c / S, scale=sigma_c / np.sqrt(S), size=[S, M])
    K = (np.ones(shape=M) * mu_K) if np.isclose(sigma_K, 0.) else np.random.normal(loc=mu_K, scale=sigma_K, size=M)
    m = (np.ones(shape=S) * mu_m) if np.isclose(sigma_m, 0.) else np.random.normal(loc=mu_m, scale=sigma_m, size=S)

    return c, K, m


def get_g(w):
    g = lambda x: np.tanh(w * x) / w
    d_g = lambda x: 1. - (np.tanh(w * x) ** 2.)
    return g, d_g

def get_f(c, K, m, w, migration):
    g, d_g = get_g(w)

    def f(t, N):
        R = K - np.dot(N, c)
        return N * (np.sum(g(c * R[np.newaxis, :]), axis=1) - m) + migration
    return f


def get_jac(c, K, m, w):
    g, d_g = get_g(w)

    def jac(t, N):
        R = K - np.dot(N, c)
        a = np.diag(np.sum(g(c * R[np.newaxis, :]), axis=1) - m)
        b = N[:, np.newaxis] * np.matmul(c * d_g(c * R[np.newaxis, :]), c.T)
        return a - b
    return jac


def integrate(c, K, m, w, t_span, migration=1e-10, N0=None, num_samples=1000):
    S = c.shape[0]
    t_eval = np.linspace(0., t_span, num=num_samples)
    N0 = np.random.rand(S) if (N0 is None) else N0

    res = solve_ivp(fun=get_f(c, K, m, w, migration),
                    t_span=(0, t_span),
                    t_eval=t_eval,
                    y0=N0,
                    atol=migration / 10.,
                    method='Radau', jac=get_jac(c, K, m, w))

    return res.t, res.y


def run_tanh_model(S, M, mu_c, sigma_c, mu_K, sigma_K, mu_m, sigma_m, w, t_span):
    c, K, m = draw_interactions(S, M, mu_c, sigma_c, mu_K, sigma_K, mu_m, sigma_m)
    return integrate(c, K, m, w, t_span)

