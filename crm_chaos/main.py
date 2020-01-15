import numpy as np
from crm_chaos.mcrm_integrator import run_mcrm
from crm_chaos.tanh_integrator import run_tanh_model


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
run_mcrm(**run_params)
'''


# MCRM with direct interaction perturbation
'''
run_params = {'S': S, 'M': M, 'mu_c': 20., 'sigma_c': 0.5,
              'mu_d': 10., 'sigma_d': 20., 'gamma': 0.,
              'mu_K': 2., 'sigma_K': 0., 'mu_m': 0.2, 'sigma_m': 0.,
              'mat_norm_ratio': 0.05, 't_span': 1e5}
run_mcrm(**run_params)
'''


# Resource-consumer model with non-linear intake function
'''
run_params = {'S': S, 'M': M, 'mu_c': 50., 'sigma_c': 1.,
              'mu_K': 5., 'sigma_K': 0., 'mu_m': 0.2, 'sigma_m': 0.,
              'w': S * 0.05, 't_span': 1e5}
run_tanh_model(**run_params)
'''


###### Chaotic phase ######

# MCRM with no direct interaction perturbation
'''
run_params = {'S': S, 'M': M, 'mu_c': 20., 'sigma_c': 4.,
              'mu_d': 10., 'sigma_d': 20., 'gamma': 0.,
              'mu_K': 2., 'sigma_K': 0., 'mu_m': 0.2, 'sigma_m': 0.,
              'mat_norm_ratio': 0., 't_span': 1e5}
run_mcrm(**run_params)
'''


# MCRM with direct interaction perturbation
'''
run_params = {'S': S, 'M': M, 'mu_c': 20., 'sigma_c': 4.,
              'mu_d': 10., 'sigma_d': 20., 'gamma': 0.,
              'mu_K': 2., 'sigma_K': 0., 'mu_m': 0.2, 'sigma_m': 0.,
              'mat_norm_ratio': 0.05, 't_span': 1e4}
run_mcrm(**run_params)
'''


# Resource-consumer model with non-linear intake function
'''
run_params = {'S': S, 'M': M, 'mu_c': 50., 'sigma_c': 15.,
              'mu_K': 5., 'sigma_K': 0., 'mu_m': 0.2, 'sigma_m': 0.,
              'w': S * 0.05, 't_span': 1e4}
run_tanh_model(**run_params)
'''