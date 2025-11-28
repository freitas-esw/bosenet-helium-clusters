
"""Single variational MC loop for BHC in JAX."""

from absl import logging

import os 
import time
import chex
import ml_collections

import jax
import jax.numpy as jnp

import numpy as np

from src import checkpoint
from src import constants
from src import networks
from src import mcmc
from src import hamiltonian
from src import writers

from src.train import AuxiliaryLossData, make_loss

from kfac_ferminet_alpha import utils as kfac_utils


def vmc(cfg: ml_collections.ConfigDict):
  """ Runs the VMC simulation in order to compute the
      main estimations of observables.

  Args:
    cfg: ConfigDict containing all necessary parameters to run the simulation.
         Check base.py for more details.
  """

  logging.info('Welcome to Bosenet Helium Cluster VMC simulation!')

  # Device logging
  num_devices = jax.device_count()
  logging.info('Starting QMC with %i XLA devices', num_devices)

  if cfg.batch_size % num_devices != 0:
    raise ValueError('Batch size must be divisible by number of devices, '
                     'got batch size {} for {} devices.'.format(
                         cfg.batch_size, num_devices))
  if cfg.system.dim != 3:
    raise ValueError('Only 3D systems are currently supported.')
  
  data_shape = (num_devices, cfg.batch_size // num_devices)

  # Set the random number generator seed
  if cfg.debug.deterministic:
    seed = 42
  else:
    seed = int(1e6 * time.time())

  key = jax.random.PRNGKey(seed)
  logging.info('RNG seed: %i', seed)


  # Restore params/data
  if not cfg.log.restore_path:
    raise ValueError('Restore path not provided!')
  ckpt_restore_path = checkpoint.get_restore_path(cfg.log.restore_path)
  ckpt_restore_filename = checkpoint.find_last_checkpoint(ckpt_restore_path)
  ckpt_save_path = ckpt_restore_path

  if ckpt_restore_filename:
    data, params, mcmc_width_ckpt = checkpoint.restore_params(ckpt_restore_filename)
    mcmc_width_ckpt = kfac_utils.replicate_all_local_devices(mcmc_width_ckpt[0,...])
    params = jax.tree_map(lambda x: x[0,...], params)
    params = kfac_utils.replicate_all_local_devices(params)
  else:
    logging.info('No checkpoint found. Stopping the simulation.')
    raise SystemExit

  key, subkey = jax.random.split(key)
  data = jnp.reshape(data, data_shape[0:] + data.shape[2:])
  data = kfac_utils.broadcast_all_local_devices(data)
  t_init = 0

  # Set up logging
  train_schema = ['step', 'energy', 'variance', 'kinetic', 'potential', 'pmove']


  # Initialisation done. We now want to have different PRNG streams on each
  # device. Shard the key over devices
  sharded_key = kfac_utils.make_different_rng_key_on_all_devices(key)


  # Main VMC simulation

  # Use the wave function without the cutoff
  batch_network = jax.vmap(networks.bosenet_vmc, (None, 0), 0)

  # Construct MCMC step
  mcmc_step = mcmc.make_mcmc_step(
          batch_network,
          cfg.batch_size // num_devices,
          steps = cfg.mcmc.steps)
  mcmc_step = constants.pmap( mcmc_step, donate_argnums=1 )


  # Construct total energy
  total_energy = make_loss(networks.bosenet_vmc, batch_network, cfg.system.interaction)
  total_energy = constants.pmap(total_energy)


  # Split the RNG keys
  sharded_key, subkeys = kfac_utils.p_split(sharded_key)


  if mcmc_width_ckpt is not None:
    mcmc_width = mcmc_width_ckpt
  else:
    mcmc_width = kfac_utils.replicate_all_local_devices(
        jnp.asarray(cfg.mcmc.width))
  
  f = open(os.path.join(ckpt_restore_path, 'samples.npy'), 'wb')

  with writers.Writer(
      name='vmc_stats',
      schema=train_schema,
      directory=ckpt_save_path,
      iteration_key=None,
      log=False) as writer:
    for t in range(t_init, cfg.vmc.iterations):
      sharded_key, subkeys = kfac_utils.p_split(sharded_key)
      data, pmove = mcmc_step(params, data, subkeys, mcmc_width)
      energy, aux = total_energy(params, data)

      # due to pmean, loss, variance and pmove should be the same across
      # devices.
      energy = energy[0]
      variance = aux.variance[0]
      kinetic = aux.kinetic[0]
      potential = aux.potential[0]
      pmove = pmove[0]

      # Update MCMC move width
      #if t < cfg.vmc.iterations // 2:
      #  if pmove > 0.55:
      #    mcmc_width *= 1.1
      #  elif pmove < 0.45:
      #    mcmc_width /= 1.1

      # Logging
      if t % cfg.log.stats_frequency == 0:
        logging.info(
            'Step %05d: %03.8f K, variance=%03.8f K^2, K=%03.8f K, V=%03.8f K, pmove=%0.2f', 
            t, energy, variance, kinetic, potential, pmove)
        writer.write(
            t,
            step=t,
            energy=np.asarray(energy),
            variance=np.asarray(variance),
            kinetic=np.asarray(kinetic),
            potential=np.asarray(potential),
            pmove=np.asarray(pmove))

      if 90*cfg.vmc.iterations <= 100*t:
        np.save(f, data)

  logging.info('The simulation finished!')

  return
