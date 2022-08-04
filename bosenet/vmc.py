
"""Single variational MC loop for BoseNet in JAX."""

from absl import logging

import chex
import ml_collections

import jax
import jax.numpy as jnp

import numpy as np

from bosenet import checkpoint
from bosenet import constants
from bosenet import networks
from bosenet import mcmc
from bosenet import hamiltonian
from bosenet.utils import writers

from kfac_ferminet_alpha import utils as kfac_utils


def init_particles(
        key,
        nparticles: int,
        ndim: int,
        batch_size: int,
        init_width: float
) -> jnp.ndarray:
  """ Initializes particles positions 

  Args:
    key: JAX RNG state
    nparticles: Number of particles 
    ndim: Number of dimensions
    batch_size: Total number of MCMC configurations to generate

  Returns:
    array (batch_size,nparticles*ndim) of initial random positions.
  """
  key, subkey = jax.random.split(key)
  positions = init_width * nparticles * jax.random.normal( subkey, shape=(batch_size,nparticles*ndim) )

  return positions


@chex.dataclass
class AuxiliaryLossData:
    """Auxiliary data returned by total_energy.

    Attributes:
      kinetic: mean kinetic energy over batch
      potential: mean potential energy over batch
      local_energy: local energy for each MCMC configuration
      variance: mean variance over batch
    """
    kinetic: jnp.DeviceArray
    potential: jnp.DeviceArray
    local_energy: jnp.DeviceArray
    variance: jnp.DeviceArray


def make_loss(network, batch_network, clip_local_energy=0.0):
  """ Creates the loss function, including custom gradients.

  Args:

  Returns:
  """
  el_fun = hamiltonian.local_energy(network)
  batch_local_energy = jax.vmap(el_fun, in_axes=(None, 0), out_axes=0)

  @jax.custom_jvp
  def total_energy(params, data):
    """ Evaluates the total energy of the network for a batch of configurations.

    Args:
    Returns:
    """
    e_l, kin, pot = batch_local_energy(params, data)
    k = constants.pmean(jnp.mean(kin)) 
    p = constants.pmean(jnp.mean(pot)) 
    loss = constants.pmean(jnp.mean(e_l)) 
    variance = constants.pmean(jnp.mean((e_l-loss)**2))
    return loss, AuxiliaryLossData(kinetic=k, potential=p, variance=variance, local_energy=e_l)

  @total_energy.defjvp
  def total_energy_jvp(primals, tangents):  # pylint: disable=unused-variable
    """ Custom Jacobian-vector product for unbiased local energy gradients. """
    params, data = primals
    loss, aux_data = total_energy(params, data)

    if clip_local_energy > 0.0:
      tv = jnp.mean(jnp.abs(aux_data.local_energy-loss))
      tv = constants.pmean(tv)
      diff = jnp.clip(aux_data.local_energy,
                      loss-clip_local_energy*tv,
                      loss+clip_local_energy*tv) - loss
    else:
      diff = aux_data.local_energy - loss

    psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)
    loss_functions.register_normal_predictive_distribution(psi_primal[:,None])
    primals_out = loss, aux_data
    tangents_out = (jnp.dot(psi_tangent, diff), aux_data)

    return primals_out, tangents_out

  return total_energy


def vmc(cfg: ml_collections.ConfigDict):
  """
  """

  logging.info('Welcome to BoseNet simulations of clusters!')

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


  # Create parameters, network, and vmapped/pmapped derivations
  key, subkey = jax.random.split(key)
  params = networks.init_bosenet_params(subkey, 
          cfg.network.hidden_dims, 
          np=cfg.system.np,
          dim=cfg.system.dim,
          num_orbitals=cfg.network.orbitals)
  params = kfac_utils.replicate_all_local_devices(params)
  batch_network = jax.vmap(networks.bosenet, (None, 0), 0)
  

# Restore params/data if necessary

  ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path)
  ckpt_restore_path = checkpoint.get_restore_path(cfg.log.restore_path)

  ckpt_restore_filename = (
      checkpoint.find_last_checkpoint(ckpt_save_path) or
      checkpoint.find_last_checkpoint(ckpt_restore_path))

  if ckpt_restore_filename:
    t_init, _, params, _, mcmc_width_ckpt = checkpoint.restore(
        ckpt_restore_filename, cfg.batch_size)
  else:
    logging.info('No checkpoint found. Stopping the simulation.')
    raise SystemExit

  key, subkey = jax.random.split(key)
  data = init_particles(
      subkey, 
      cfg.system.np, 
      cfg.system.dim, 
      cfg.batch_size, 
      cfg.mcmc.init_width)
  data = jnp.reshape(data, data_shape + data.shape[1:])
  data = kfac_utils.broadcast_all_local_devices(data)
  t_init = 0

  # Set up logging
  train_schema = ['step', 'energy', 'variance', 'kinetic', 'potential', 'pmove']


  # Initialisation done. We now want to have different PRNG streams on each
  # device. Shard the key over devices
  sharded_key = kfac_utils.make_different_rng_key_on_all_devices(key)


  # Main training

  # Construct MCMC step
  mcmc_step = mcmc.make_mcmc_step(
          batch_network,
          cfg.batch_size // num_devices,
          steps = cfg.mcmc.steps)
  mcmc_step = constants.pmap( mcmc_step, donate_argnums=1 )


  # Construct total energy
  total_energy = make_loss(networks.bosenet, batch_network)
  total_energy = constants.pmap(total_energy)


  # Split the RNG keys
  sharded_key, subkeys = kfac_utils.p_split(sharded_key)


  if mcmc_width_ckpt is not None:
    mcmc_width = mcmc_width_ckpt
  else:
    mcmc_width = kfac_utils.replicate_all_local_devices(
        jnp.asarray(cfg.mcmc.width))
  

  if t_init == 0:
    logging.info('Burning in MCMC chain for %d steps', cfg.mcmc.burn_in)
    for t in range(cfg.mcmc.burn_in):
      sharded_key, subkeys = kfac_utils.p_split(sharded_key)
      data, pmove = mcmc_step(params, data, subkeys, mcmc_width)
    logging.info('Completed burn-in MCMC steps')
    logging.info('Initial energy: %03.4f K', total_energy(params, data)[0][0])


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
      if t < cfg.vmc.iterations // 2:
        if pmove > 0.55:
          mcmc_width *= 1.1
        elif pmove < 0.45:
          mcmc_width /= 1.1


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


  logging.info('The simulation finished!')

  return
