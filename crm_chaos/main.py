import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from crm_chaos.mcrm_integrator import run_mcrm
from crm_chaos.tanh_integrator import run_tanh_model


def plot_dynamics(t, N, species_count=20, mid_species_count=10, title=None):
    N = N[np.argsort(N[:, -1]), :]  # Sort species by abundance at the last time point
    N_count = N.shape[0]

    #Pick species to plot uniformly along species abundance range
    uniform_range = np.arange(0, N_count, N_count / species_count, dtype=np.int)

    # Pick species to plot around mid species abundance
    final_log_N = np.log10(N[:, -1])
    mid_N = (final_log_N.max() + final_log_N.min()) / 2.
    mid_idx = np.argmin(np.abs(final_log_N - mid_N))
    mid_range = np.arange(mid_idx - (mid_species_count / 2), mid_idx + (mid_species_count / 2), dtype=np.int)

    fig = plt.figure()
    axes = fig.gca()

    for i in np.concatenate([uniform_range, mid_range]):
        axes.semilogy(t, N[i, :])

    plt.xlabel('Time')
    plt.ylabel('N, Species abundance')
    if title is not None:
        plt.title(title)


'''
Examples for time evolution of consumer-resource models in fixed point and chaotic phases.
Uncomment single section and run.
'''

S = 800
M = 60


###### Fixed point phase ######

# MCRM with no direct interaction perturbation
'''
run_params = {'S': S, 'M': M, 'mu_c': 20., 'sigma_c': 0.5,
              'mu_d': 10., 'sigma_d': 20., 'gamma': 0.,
              'mu_K': 2., 'sigma_K': 0., 'mu_m': 0.2, 'sigma_m': 0.,
              'mat_norm_ratio': 0., 't_span': 1e5}
t, N = run_mcrm(**run_params)
plot_dynamics(t, N, title='FP phase - without direct interactions pert.')
'''


# MCRM with direct interaction perturbation
run_params = {'S': S, 'M': M, 'mu_c': 20., 'sigma_c': 0.5,
              'mu_d': 10., 'sigma_d': 20., 'gamma': 0.,
              'mu_K': 2., 'sigma_K': 0., 'mu_m': 0.2, 'sigma_m': 0.,
              'mat_norm_ratio': 0.05, 't_span': 1e5}
t, N = run_mcrm(**run_params)
plot_dynamics(t, N, title='FP phase - with direct interactions pert.')


# Resource-consumer model with non-linear intake function
'''
run_params = {'S': S, 'M': M, 'mu_c': 50., 'sigma_c': 1.,
              'mu_K': 5., 'sigma_K': 0., 'mu_m': 0.2, 'sigma_m': 0.,
              'w': S * 0.05, 't_span': 1e5}
t, N = run_mcrm(**run_params)
plot_dynamics(t, N, title='FP phase - non-linear intake function')
'''


###### Chaotic phase ######

# MCRM with no direct interaction perturbation
'''
run_params = {'S': S, 'M': M, 'mu_c': 20., 'sigma_c': 4.,
              'mu_d': 10., 'sigma_d': 20., 'gamma': 0.,
              'mu_K': 2., 'sigma_K': 0., 'mu_m': 0.2, 'sigma_m': 0.,
              'mat_norm_ratio': 0., 't_span': 1e5}
t, N = run_mcrm(**run_params)
plot_dynamics(t, N, title='Chaotic phase - without direct interactions pert.')
'''


# MCRM with direct interaction perturbation
run_params = {'S': S, 'M': M, 'mu_c': 20., 'sigma_c': 4.,
              'mu_d': 10., 'sigma_d': 20., 'gamma': 0.,
              'mu_K': 2., 'sigma_K': 0., 'mu_m': 0.2, 'sigma_m': 0.,
              'mat_norm_ratio': 0.05, 't_span': 1e4}
t, N = run_mcrm(**run_params)
plot_dynamics(t, N, title='Chaotic phase - with direct interactions pert.')


# Resource-consumer model with non-linear intake function
'''
run_params = {'S': S, 'M': M, 'mu_c': 50., 'sigma_c': 15.,
              'mu_K': 5., 'sigma_K': 0., 'mu_m': 0.2, 'sigma_m': 0.,
              'w': S * 0.05, 't_span': 1e4}
t, N = run_mcrm(**run_params)
plot_dynamics(t, N, title='Chaotic phase - non-linear intake function')
'''