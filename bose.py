
"""Main wrapper for BoseNet in JAX."""

from absl import app
from absl import flags
from absl import logging
from bosenet import base_config
from bosenet import train
from ml_collections.config_flags import config_flags

from jax.config import config
config.update("jax_enable_x64", True)

# internal imports

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config', None, 'Path to config file.')


def main(_):
  cfg = FLAGS.config
  logging.info('System config:\n\n%s', cfg)
  train.train(cfg)


if __name__ == '__main__':
  app.run(main)
