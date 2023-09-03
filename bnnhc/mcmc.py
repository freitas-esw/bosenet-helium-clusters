
"""Metropolis-Hastings Monte Carlo.

NOTE: these functions operate on batches of MCMC configurations and should not
be vmapped.
"""

from bnnhc import constants
import jax
from jax import lax
from jax import numpy as jnp


def mh_update(
    params,
    f,
    x1,
    key,
    lp_1,
    num_accepts,
    stddev=0.02
):
  """ Performs one Metropolis-Hastings step using an all-electron move
    
  Args:
    params: variational parameters
    f: function with signature f(params, x) that computes the log of wave function
    x1: current set of particles positions
    key: RNG state
    lp_1: current log probabilities of f evaluated at x1
    num_accepts: number of accepted trial moves
    stddev: width of Gaussian move proposal 

  Returns:
    ( x, key, lp, num_accepts ), where:
      x: Updated set of particle positions
      key: RNG state
      lp: function f evaluated at x
      num_accepts: updated total number of accepted moves
  """
  key, subkey = jax.random.split(key)
  x2 = x1 + stddev * jax.random.normal( subkey, shape=x1.shape )
  lp_2 = 2. * f( params, x2 )
  ratio = lp_2 - lp_1

  key, subkey = jax.random.split(key)
  rnd = jnp.log( jax.random.uniform( subkey, shape=lp_1.shape ) )
  cond = ratio > rnd
  x_new = jnp.where( cond[...,None], x2, x1 )
  lp_new = jnp.where( cond, lp_2, lp_1 )
  num_accepts += jnp.sum(cond)

  return x_new, key, lp_new, num_accepts


def make_mcmc_step(
    batch_network,
    batch_per_device,
    steps=10
):
  """ Creates the MCMC step function.

  Args:
    batch_network: vectorised function that computes the log wave function
    batch_per_device: batch size per device
    steps: number of trial moves to attempt
  """

  @jax.jit
  def mcmc_step(params, data, key, width):
    """ Performs a set of MCMC steps.
    
    Args:
      params: set of variational parameters
      data: set of particle positions
      key: RNG key
      width: standard deviation for move proposals

    Returns:
      (data, pmove): data is the updated positions and pmove the average 
                     probability a move was accepted
    """

    def step_fn(i, x):
      return mh_update(params, batch_network, *x, stddev=width)

    logprob = 2.*batch_network(params, data)
    data, key, _, num_accepts = lax.fori_loop(0, steps, step_fn,
                                               (data, key, logprob, 0.))
    pmove = jnp.sum(num_accepts) / (steps*batch_per_device)
    pmove = constants.pmean(pmove)

    return data, pmove

  return mcmc_step

