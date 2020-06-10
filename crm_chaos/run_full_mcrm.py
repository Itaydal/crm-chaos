import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from crm_chaos.full_mcrm_integrator import run_full_mcrm


def plot_dynamics_single(t, y, ax, ylabel, species_count=20, mid_species_count=10):
    y = y[np.argsort(y[:, -1]), :]  # Sort species by abundance at the last time point
    y_count = y.shape[0]

    #Pick species to plot uniformly along species abundance range
    uniform_range = np.arange(0, y_count, y_count / species_count, dtype=np.int)

    # Pick species to plot around mid species abundance
    final_log_y = np.log10(y[:, -1])
    mid_y = (final_log_y.max() + final_log_y.min()) / 2.
    mid_idx = np.argmin(np.abs(final_log_y - mid_y))
    mid_range = np.arange(mid_idx - (mid_species_count / 2), mid_idx + (mid_species_count / 2), dtype=np.int)

    for i in np.concatenate([uniform_range, mid_range]):
        ax.semilogy(t, y[i, :])

    ax.set(xlabel='Time', ylabel=ylabel)


def plot_dynamics(t, N, R, species_count=20, mid_species_count=10, title=None):
    fig, axs = plt.subplots(2)
    fig.suptitle(title)

    plot_dynamics_single(t, N, axs[0], 'N, Species abundance',
                         species_count=species_count, mid_species_count=mid_species_count)
    plot_dynamics_single(t, R, axs[1], 'R, Resource availability',
                         species_count=species_count, mid_species_count=mid_species_count)


'''
Examples for time evolution of consumer-resource models in fixed point and chaotic phases.
Here we simulate the consumer-resource model without assuming resource dynamics is at equilibrium.
Therefore there are distinct dynamic variables for species and resources.
In this case, it is guaranteed that resource availabilities remain non-negative throughout the dynamics.
Moreover, after drawing the consumption preference matrix c, negative entries are set to zero.
These simulations show that the same phenomena described in the paper also appear in a naturally realistic example.   
  
Uncomment single section and run.
'''

S = 800
M = 120


###### Fixed point phase ######

'''
run_params = {'S': S, 'M': M, 'mu_c': 50., 'sigma_c': 0.5,
              'mu_d': 10., 'sigma_d': 2., 'gamma': 0.,
              'mu_K': 2., 'sigma_K': 0., 'mu_m': 0.2, 'sigma_m': 0.,
              'kappa': 0.,'mat_norm_ratio': 0.05, 't_span': 1e5}
t, N, R = run_full_mcrm(**run_params)
plot_dynamics(t, N, R title='Full CRM at the fixed point phase')
'''


###### Chaotic phase ######

run_params = {'S': S, 'M': M, 'mu_c': 50., 'sigma_c': 3.,
              'mu_d': 10., 'sigma_d': 2., 'gamma': 0.,
              'mu_K': 2., 'sigma_K': 0., 'mu_m': 0.2, 'sigma_m': 0.,
              'kappa': 0., 'mat_norm_ratio': 0.05, 't_span': 1e4}
t, N, R = run_full_mcrm(**run_params)
plot_dynamics(t, N, R, title='Full CRM at the chaotic phase')



###### Unbounded growth phase ######

'''
By adding small regularizing term to the growth rate in the form of (-kappa * N_i^2) and setting kappa=1e-2
we find that the diverging behavior of the model vanish, and chaotic phase extend instead.
'''

# Resource-consumer model with kappa=0 at the diverging phase
'''
run_params = {'S': S, 'M': M, 'mu_c': 50., 'sigma_c': 6.,
              'mu_d': 10., 'sigma_d': 2., 'gamma': 0.,
              'mu_K': 2., 'sigma_K': 0., 'mu_m': 0.2, 'sigma_m': 0.,
              'kappa': 0., 'mat_norm_ratio': 0.05, 't_span': 1e4, 'num_time_evals': None}
t, N, R = run_full_mcrm(**run_params)
plot_dynamics(t, N, R, title='Full CRM at the unbounded growth phase')
'''


# Resource-consumer model with kappa=1e-2 for the same parameters
'''
run_params = {'S': S, 'M': M, 'mu_c': 50., 'sigma_c': 6.,
              'mu_d': 10., 'sigma_d': 2., 'gamma': 0.,
              'mu_K': 2., 'sigma_K': 0., 'mu_m': 0.2, 'sigma_m': 0.,
              'kappa': 1e-2, 'mat_norm_ratio': 0.05, 't_span': 1e4}
t, N, R = run_full_mcrm(**run_params)
plot_dynamics(t, N, R, title='Full CRM at the unbounded growth phase - with regularization')
'''