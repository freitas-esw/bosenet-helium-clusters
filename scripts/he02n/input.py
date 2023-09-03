
from bhc import base_config


def get_config():
  """Returns config for running helium clusters with qmc."""
  cfg = base_config.default()

  cfg.debug.deterministic = True

  cfg.batch_size = 2048

  cfg.system.np = 2

  cfg.optim.iterations = 2000
  cfg.optim.lr.rate = 0.01
  cfg.optim.lr.delay = 2000

  cfg.mcmc.width = 1.4
  cfg.mcmc.burn_in = 100

  cfg.log.save_frequency = 2000

  cfg.log.save_path = 'scripts/he02n'
  cfg.log.restore_path = 'scripts/he02n'

  cfg.network.hidden_dims = ((32,16),(32,16),(32,16),(32,16))
  cfg.network.orbitals = 2
  
  cfg.vmc.block_size = 2048
  cfg.vmc.iterations = 2000

  return cfg
