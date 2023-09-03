
"""Generic Helium cluster configuration for BoseNet."""

from bosenet import base_config


def get_config():
  """Returns config for running generic helium clusters with qmc."""
  cfg = base_config.default()
  return cfg
