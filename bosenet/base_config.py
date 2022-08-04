
"""Default base configuration for helium cluster calculations."""

import enum

import ml_collections
from ml_collections import config_dict


def default() -> ml_collections.ConfigDict:
  """Create set of default parameters for running train.py.

  Returns:
    ml_collections.ConfigDict containing default settings.
  """
  # wavefunction output.
  cfg = ml_collections.ConfigDict({
      'batch_size': 8192,  # batch size

      # Do *not* override on command-line. Do *not* set using __name__ from 
      # inside a get_config function, as config_flags overrides this when 
      # importing the module using importlib.import_module.
      'config_module': __name__,

      'optim': {
          'iterations': 200000,   # number of iterations
          'lr': {
              'rate': 1.e-4,     # learning rate
              'decay': 1.0,      # exponent of learning rate decay
              'delay': 10000.0,  # term that sets the scale of the rate decay
          },
          'clip_el': 0.0,        # if not none, scale at which to clip local energy
          
          # KFAC hyperparameters. See KFAC documentation for details.
          'kfac': {
              'invert_every': 1,
              'cov_update_every': 1,
              'damping': 0.001,
              'cov_ema_decay': 0.95,
              'momentum': 0.0,
              'momentum_type': 'regular',
              # Warning: adaptive damping is not currently available.
              'min_damping': 1.e-4,
              'norm_constraint': 0.001,
              'mean_center': True,
              'l2_reg': 0.0,
              'register_only_generic': False,
          },
      },

      'log': {
          'stats_frequency': 1,    # iterations between logging of stats
          'save_frequency': 20000, # steps between saving network params
          'save_path': '',         # path to save/restore network to/from
          'restore_path': '',      # path containing checkpoint to restore network from
      },

      'system': {
          'np': 3,   # Number of particles in system
          'dim': 3,  # Number of system dimensions  
      },

      'mcmc': {
          'burn_in': 100,        # number of burn in steps before optimization
          'steps': 10,           # number of MCMC steps to make between network updates
          'init_width': 1.20,    # width of Gaussian used to generate initial configurations
          'width': 0.10,         # width of Gaussian used for random moves step size for
          'adapt_frequency': 25, # number of steps after which to update the MCMC step size
      },

      'network': { 
         'hidden_dims': ((256,32),(256,32),(256,32),(256,32)),
         'orbitals': 16,
      },

      'vmc':{
          'block_size': 4096, # size of the VMC blocks 
          'iterations': 4000, # number of VMC iterations
      },

      'debug': {
          'check_nan': False,      # check loss and parameter 
          'deterministic': False,  # use a deterministic seed
      },


  })

  return cfg
