
"""Core training loop for BHC-NN QMC in JAX."""

import time

import ml_collections
import chex

from absl import logging

import jax
import jax.numpy as jnp
import numpy as np

from src import checkpoint
from src import utils
from src import networks
from src import mcmc
from src import hamiltonian
from src import writers
from src import curvature_tags_and_blocks

import kfac_jax
#from kfac_ferminet_alpha import loss_functions
#from kfac_ferminet_alpha import utils as kfac_utils
#from kfac_ferminet_alpha import optimizer as kfac_optim


def init_particles(
        key,
        nparticles: int,
        ndim: int,
        batch_size: int,
        init_width: float
) -> jnp.ndarray:
  """ Initializes particles positions.

  Args:
    key: JAX RNG state
    nparticles: number of particles 
    ndim: number of spatial dimensions
    batch_size: total number of configurations to generate

  Returns:
    array (batch_size, nparticles*ndim) of initial random positions.
  """
  key, subkey = jax.random.split(key)
  a = init_width * float(nparticles)**(1.0/3.0)
  positions = a * jax.random.normal( subkey, shape=(batch_size,nparticles*ndim) )

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
    kinetic: jax.Array
    potential: jax.Array
    local_energy: jax.Array
    variance: jax.Array


def make_loss(network, batch_network, pot_type='aziz'):
  """ Creates the loss function, including custom gradients.

  Args:
    network: function that computes the log trial wave function for a single
             set of particle positions
    batch_network: same as network but for a batch of set of positions
    pot_type: which potential to use, either HFD-He (aziz) or Lennard-Jones (lj)

  Returns:
    total_energy: function that estimates the total energy through MC integration
  """
  el_fun = hamiltonian.local_energy(network, pot_type)
  batch_local_energy = jax.vmap(el_fun, in_axes=(None, 0), out_axes=0)

  @jax.custom_jvp
  def total_energy(params, data):
    """ Evaluates the total energy of the network for a batch of configurations.

    Args:
      params: variational parameters
      data: a batch of a set of particles positions

    Returns:
      loss: total energy estimation
      aux_data: estimated kinetic and potential energies, as well as the local energies and
                the total energy variance
    """
    e_l, kin, pot = batch_local_energy(params, data)
    k = utils.pmean(jnp.mean(kin)) 
    p = utils.pmean(jnp.mean(pot)) 
    loss = utils.pmean(jnp.mean(e_l)) 
    variance = utils.pmean(jnp.mean((e_l-loss)**2))
    return loss, AuxiliaryLossData(kinetic=k, potential=p, variance=variance, local_energy=e_l)

  @total_energy.defjvp
  def total_energy_jvp(primals, tangents):  # pylint: disable=unused-variable
    """ Custom Jacobian-vector product for unbiased local energy gradients. """
    
    params, data = primals
    loss, aux_data = total_energy(params, data)

    diff = aux_data.local_energy - loss
    
    psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)
    #loss_functions.register_normal_predictive_distribution(psi_primal[:,None])
    kfac_jax.register_normal_predictive_distribution(psi_primal[:, None])
    primals_out = loss, aux_data
    tangents_out = (jnp.dot(psi_tangent, diff), aux_data)

    return primals_out, tangents_out

  return total_energy


def train(cfg: ml_collections.ConfigDict):
  """ Runs the main simulation in order to optimise the
      variational parameters.

  Args:
    cfg: ConfigDict containing all necessary parameters to run the simulation.
         Check base.py for more details.
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
    seed = cfg.debug.seed
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
  #params = kfac_utils.replicate_all_local_devices(params)
  params = utils.replicate(params)
  batch_network = jax.vmap(networks.bosenet, (None, 0), 0)
  

# Set up checkpointing and restore params/data if necessary

  ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path)
  ckpt_restore_path = checkpoint.get_restore_path(cfg.log.restore_path)

  ckpt_restore_filename = (
      checkpoint.find_last_checkpoint(ckpt_save_path) or
      checkpoint.find_last_checkpoint(ckpt_restore_path))

  if ckpt_restore_filename:
    t_init, data, params, opt_state_ckpt, mcmc_width_ckpt = checkpoint.restore(
        ckpt_restore_filename, cfg.batch_size)
    if ckpt_restore_path:
      t_init = 0 if ckpt_save_path != ckpt_restore_path else t_init
  else:
    logging.info('No checkpoint found. Training new model.')
    key, subkey = jax.random.split(key)
    data = init_particles(
        subkey, 
        cfg.system.np, 
        cfg.system.dim, 
        cfg.batch_size, 
        cfg.mcmc.init_width)
    data = jnp.reshape(data, data_shape + data.shape[1:])
    #data = kfac_utils.broadcast_all_local_devices(data)
    data = utils.broadcast(data)
    t_init = 0
    opt_state_ckpt = None
    mcmc_width_ckpt = None

  # Set up logging
  train_schema = ['step', 'energy', 'variance', 'kinetic', 'potential', 'pmove']


  # Initialisation done. We now want to have different PRNG streams on each
  # device. Shard the key over devices
  #sharded_key = kfac_utils.make_different_rng_key_on_all_devices(key)
  sharded_key = utils.shard_key(key)


  # Main training

  # Construct MCMC step
  mcmc_step = mcmc.make_mcmc_step(
          batch_network,
          cfg.batch_size // num_devices,
          steps = cfg.mcmc.steps)


  # Construct loss and optimizer
  total_energy = make_loss(networks.bosenet, batch_network, cfg.system.interaction)


  # Compute the learning rate
  def learning_rate_schedule(t):
    return cfg.optim.lr.rate*jnp.power(
            (1.0 / (1.0+(t/cfg.optim.lr.delay))), cfg.optim.lr.decay)


  # Differentiate wrt parameters (argument 0)
  val_n_grad = jax.value_and_grad( total_energy, argnums=0, has_aux=True )


  # Set up the KFAC optimizer
  #optimizer = kfac_optim.Optimizer(
  optimizer = kfac_jax.Optimizer(
      val_n_grad,
      l2_reg=cfg.optim.kfac.l2_reg,
      norm_constraint=cfg.optim.kfac.norm_constraint,
      value_func_has_aux=True,
      value_func_has_rng=False,
      learning_rate_schedule=learning_rate_schedule,
      curvature_ema=cfg.optim.kfac.cov_ema_decay,
      inverse_update_period=cfg.optim.kfac.invert_every,
      min_damping=cfg.optim.kfac.min_damping,
      num_burnin_steps=0,
      register_only_generic=cfg.optim.kfac.register_only_generic,
      estimation_mode='fisher_exact',
      multi_device=True,
      pmap_axis_name=utils.PMAP_AXIS_NAME,
      auto_register_kwargs=dict(
          graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,)
      # debug=True
  )

  #sharded_key, subkeys = kfac_utils.p_split(sharded_key)
  sharded_key, subkeys = utils.p_split(sharded_key)
  opt_state = optimizer.init(params, subkeys, data)
  opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state

  mcmc_step = utils.pmap( mcmc_step, donate_argnums=1 )

  if mcmc_width_ckpt is not None:
    mcmc_width = mcmc_width_ckpt
  else:
    #mcmc_width = kfac_utils.replicate_all_local_devices(
    mcmc_width = utils.replicate(
        jnp.asarray(cfg.mcmc.width))
  
  pmoves = np.zeros(cfg.mcmc.adapt_frequency)
  #shared_t = kfac_utils.replicate_all_local_devices(jnp.zeros([]))
  shared_t = utils.replicate(jnp.zeros([]))
  #shared_mom = kfac_utils.replicate_all_local_devices(jnp.zeros([]))
  shared_mom = utils.replicate(jnp.zeros([]))
  #shared_damping = kfac_utils.replicate_all_local_devices(
  shared_damping = utils.replicate(
      jnp.asarray(cfg.optim.kfac.damping))
  

  if t_init == 0:
    logging.info('Burning in MCMC chain for %d steps', cfg.mcmc.burn_in)
    for t in range(cfg.mcmc.burn_in):
      #sharded_key, subkeys = kfac_utils.p_split(sharded_key)
      sharded_key, subkeys = utils.p_split(sharded_key)
      data, pmove = mcmc_step(params, data, subkeys, mcmc_width)
    logging.info('Completed burn-in MCMC steps')
    logging.info('Initial energy: %03.4f K',
                 utils.pmap(total_energy)(params, data)[0][0])

  time_of_last_ckpt = time.time()


  with writers.Writer(
      name='train_stats',
      schema=train_schema,
      directory=ckpt_save_path,
      iteration_key=None,
      log=False) as writer:
    for t in range(t_init, cfg.optim.iterations):
      #sharded_key, subkeys = kfac_utils.p_split(sharded_key)
      sharded_key, subkeys = utils.p_split(sharded_key)
      data, pmove = mcmc_step(params, data, subkeys, mcmc_width)
      # Need this split because MCMC step above used subkeys already
      #sharded_key, subkeys = kfac_utils.p_split(sharded_key)
      sharded_key, subkeys = utils.p_split(sharded_key)
      params, opt_state, stats = optimizer.step(  # pytype: disable=attribute-error
          params=params,
          state=opt_state,
          rng=subkeys,
          data_iterator=iter([data]),
          momentum=shared_mom,
          damping=shared_damping)
      loss = stats['loss']
      aux_data = stats['aux']

      # due to pmean, loss, variance and pmove should be the same across
      # devices.
      loss = loss[0]
      variance = aux_data.variance[0]
      kinetic = aux_data.kinetic[0]
      potential = aux_data.potential[0]
      pmove = pmove[0]

      # Update MCMC move width
      if t > 0 and t % cfg.mcmc.adapt_frequency == 0:
        if np.mean(pmoves) > 0.50:
          mcmc_width *= 1.05
        if np.mean(pmoves) < 0.45:
          mcmc_width /= 1.05
        pmoves[:] = 0
      else:
        if pmove > 0.999:
          mcmc_width *= 1.1
        elif pmove < 0.199:
          mcmc_width /= 1.1
      pmoves[t%cfg.mcmc.adapt_frequency] = pmove

      if cfg.debug.check_nan:
        tree = {'params': params, 'loss': loss}
        if cfg.optim.optimizer != 'none':
          tree['optim'] = opt_state
        chex.assert_tree_all_finite(tree)

      # Logging
      if t % cfg.log.stats_frequency == 0:
        logging.info(
            'Step %05d: %03.8f K, variance=%03.8f K^2, K=%03.8f K, V=%03.8f K, pmove=%0.2f', 
            t, loss, variance, kinetic, potential, pmove)
        writer.write(
            t,
            step=t,
            energy=np.asarray(loss),
            variance=np.asarray(variance),
            kinetic=np.asarray(kinetic),
            potential=np.asarray(potential),
            pmove=np.asarray(pmove))

      if (t+1) % cfg.log.save_frequency == 0:
        checkpoint.save(ckpt_save_path, t, data, params, opt_state, mcmc_width)
        time_of_last_ckpt = time.time()

  try:
    if (t+1) % cfg.log.save_frequency != 0:
      checkpoint.save(ckpt_save_path, t, data, params, opt_state, mcmc_width)
  except: 
    logging.info('Warning: variable t_init larger than cfg.optim.iterations')

  logging.info('The simulation finished!')

  return
