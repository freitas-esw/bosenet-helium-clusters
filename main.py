
"""Main wrapper for Bosonic Neural Networks - Helium Clusters (BNN-HC) in JAX."""

from absl import app
from absl import flags
from absl import logging
from src import base
from src import train
from src import vmc
from ml_collections.config_flags import config_flags

from jax import config
config.update("jax_enable_x64", True)

# internal imports

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config', None, 'Path to config file.')


def main(_):

  cfg = FLAGS.config

  # Log the configuration of the simulation
  logging.info('System config:\n\n%s', cfg)
  
  # Greetings to start the simulation
  logging.info('Welcome to Python environment for Monte Carlo Simulations!')

  if cfg.method == 'train':
    train.train(cfg)
  elif cfg.method == 'vmc':
    vmc.vmc(cfg)
  else:
    raise ValueError(f'Unknown method {cfg.method}')


if __name__ == '__main__':
  app.run(main)
